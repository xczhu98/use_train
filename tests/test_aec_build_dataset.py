"""Tests for tasks.aec.data.build_dataset config handling."""
import pytest

from unittest.mock import patch

import tasks.aec.data as aec_data
from tasks.aec.data import build_dataset


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
    with patch("tasks.aec.data.DynamicMixingDataset") as mock_ds:
        build_dataset(cfg["data"], mode="train")
        build_dataset(cfg["data"], mode="val", meta=cfg.get("meta"))
    val_kwargs = mock_ds.call_args.kwargs
    assert val_kwargs["max_duration_seconds"] == 10.0
    assert val_kwargs["repeat_to_max_duration"] is True
    # Original cfg must remain intact.
    assert cfg["data"]["train"]["max_duration_seconds"] == 10.0
    assert cfg["data"]["train"]["repeat_to_max_duration"] is True


def test_compatibility_api_delegates_with_aec_build_spec():
    """The retained public wrapper must keep its AEC-specific shared-builder wiring."""
    sentinel = object()
    cfg = {"mode": "dynamic"}
    meta = {"seed": 7}

    with patch.object(aec_data, "_build_dataset_impl", return_value=sentinel) as impl:
        result = aec_data.build_dataset(cfg, mode="val", meta=meta)

    assert result is sentinel
    assert impl.call_count == 1
    call = impl.call_args
    assert call.args == (cfg,)
    assert call.kwargs["mode"] == "val"
    assert call.kwargs["meta"] is meta
    spec = call.kwargs["spec"]
    assert spec.task == "aec"
    assert spec.dynamic_dataset_cls is aec_data.DynamicMixingDataset


pytestmark = [pytest.mark.unit]
