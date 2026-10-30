"""Tests for tasks.se.data.build_dataset config handling."""
import pytest

from unittest.mock import patch

from tasks.se.data import build_dataset


def _base_cfg():
    return {
        "data": {
            "mode": "dynamic",
            "sample_rate": 16000,
            "train": {
                "speech_scp": "dummy.scp",
                "noise_scp": "dummy.scp",
                "max_duration_seconds": 10.0,
                "repeat_to_max_duration": True,
                "simulation_cfg": {},
            },
            "validation": {
                "mode": "dynamic",
                "inherit_simulation": True,
                "speech_scp": "dummy_val.scp",
            },
            "dataloader": {"batch_size": 1, "num_workers": 0},
        },
        "meta": {"seed": 0},
    }


def test_val_inherits_layer_a_keys_after_train_build():
    """Regression: building train must not mutate cfg in place, otherwise val
    inherits a stripped simulation_cfg and loses max_duration_* settings."""
    cfg = _base_cfg()
    with patch("tasks.se.data.DynamicMixingDataset") as mock_ds:
        build_dataset(cfg["data"], mode="train")
        build_dataset(cfg["data"], mode="val", meta=cfg.get("meta"))
    val_kwargs = mock_ds.call_args.kwargs
    assert val_kwargs["max_duration_seconds"] == 10.0
    assert val_kwargs["repeat_to_max_duration"] is True
    # Original cfg must remain intact.
    assert cfg["data"]["train"]["max_duration_seconds"] == 10.0
    assert cfg["data"]["train"]["repeat_to_max_duration"] is True


def test_pase_preprocess_is_projected_from_simulation_cfg_to_dataset_kwarg():
    """PASE 专属预处理整块从 simulation_cfg 提升为 Dataset 构造参数。"""
    cfg = _base_cfg()
    cfg["data"]["train"]["simulation_cfg"] = {
        "pase_preprocess": {
            "peak_normalization_mode": "separate",
            "peak_normalization_scale_range": [0.5, 0.95],
        }
    }
    with patch("tasks.se.data.DynamicMixingDataset") as mock_ds:
        build_dataset(cfg["data"], mode="train")
    assert mock_ds.call_args.kwargs["pase_preprocess"] == {
        "peak_normalization_mode": "separate",
        "peak_normalization_scale_range": [0.5, 0.95],
    }


@pytest.mark.parametrize(
    "pase_preprocess",
    [
        {"apply_after_channel": True},
        {"apply_after_channel": False, "peak_normalization_mode": "separate"},
    ],
    ids=("deferred-only", "explicit-false"),
)
def test_pase_preprocess_apply_after_channel_is_projected_to_dataset_kwarg(
    pase_preprocess,
):
    """apply_after_channel 与既有 PASE 键一样整块投影到 Dataset 构造参数。"""
    cfg = _base_cfg()
    cfg["data"]["train"]["simulation_cfg"] = {"pase_preprocess": pase_preprocess}
    with patch("tasks.se.data.DynamicMixingDataset") as mock_ds:
        build_dataset(cfg["data"], mode="train")
    assert mock_ds.call_args.kwargs["pase_preprocess"] == pase_preprocess


@pytest.mark.parametrize(
    ("pase_preprocess", "message"),
    [
        ({"peak_mode": "separate"}, "unknown keys"),
        ({"peak_normalization_mode": "joint"}, "peak_normalization_mode"),
        ({"peak_normalization_scale_range": [0.5]}, "2-element sequence"),
        ({"peak_normalization_scale_range": "0.5,0.95"}, "2-element sequence"),
        ({"peak_normalization_scale_range": ["a", "b"]}, "two finite numbers"),
        ({"peak_normalization_scale_range": [0.9, 0.5]}, "0 < lo <= hi <= 1"),
        ({"peak_normalization_scale_range": [-0.5, 0.9]}, "0 < lo <= hi <= 1"),
        ({"peak_normalization_scale_range": [0.5, 1.5]}, "0 < lo <= hi <= 1"),
        ({"peak_normalization_scale_range": [float("nan"), 0.9]}, "0 < lo <= hi <= 1"),
        ({"apply_after_channel": 1}, "apply_after_channel.*exact bool"),
        ({"apply_after_channel": "true"}, "apply_after_channel.*exact bool"),
        ({"apply_after_channel": None}, "apply_after_channel.*exact bool"),
    ],
)
def test_dynamic_dataset_rejects_invalid_pase_preprocess(pase_preprocess, message):
    """配置边界 fail-fast：非法键与非法 scale range 不得留到采样期。"""
    from tasks.se.data.dynamic_dataset import DynamicMixingDataset

    with pytest.raises(ValueError, match=message):
        DynamicMixingDataset(
            speech_scp="missing.scp",
            noise_scp="missing.scp",
            rir_scp="missing.scp",
            pase_preprocess=pase_preprocess,
        )


@pytest.mark.parametrize(
    "scale_range",
    [[0.5], [], [0.5, 0.9, 0.95], ["a", "b"]],
    ids=("single", "empty", "three", "non-numeric"),
)
def test_pase_peak_preprocess_rejects_malformed_scale_range(scale_range):
    """采样期入口不得对残缺 range 抛 IndexError / 静默取前两项。"""
    import numpy as np

    from tasks.se.data.pase_preprocess import apply_pase_peak_preprocess

    waveform = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="scale_range"):
        apply_pase_peak_preprocess(
            waveform,
            waveform,
            waveform,
            mode="separate",
            scale_range=scale_range,
            rng=np.random.default_rng(0),
        )


pytestmark = [pytest.mark.unit]
