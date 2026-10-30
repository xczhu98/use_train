"""Spatial Dataset 与模型 microphone-axis geometry 合同。"""

from types import SimpleNamespace

import pytest

from core.data.simulation_configs import SimulationConfigs
from core.models.io_contract import (
    ModelIOContract,
    TensorField,
    validate_spatial_model_geometry,
)


pytestmark = pytest.mark.unit


def _dataset(reference_channel=1):
    return SimpleNamespace(
        sim_cfg=SimulationConfigs.from_dict(
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": reference_channel,
            },
            strict=True,
        )
    )


class _MonoModel:
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"se"}),
        inputs=(TensorField("noisy_spec", "spectrum", ("batch", "frequency", "frame", "complex")),),
        outputs=(TensorField("estimate", "spectrum", ("batch", "frequency", "frame", "complex")),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )


class _SpatialModel:
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"se"}),
        inputs=(TensorField("noisy_spec", "spectrum", ("batch", "microphone", "frequency", "frame", "complex")),),
        outputs=(TensorField("estimate", "spectrum", ("batch", "frequency", "frame", "complex")),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def __init__(self, n_channel=2, ref_channel=1):
        self.n_channel = n_channel
        self.ref_channel = ref_channel


def test_spatial_dataset_rejects_mono_model_at_build_boundary():
    with pytest.raises(ValueError, match="native microphone-axis"):
        validate_spatial_model_geometry(_MonoModel(), _dataset())


def test_spatial_model_geometry_accepts_matching_channels_and_reference():
    validate_spatial_model_geometry(_SpatialModel(), _dataset())


def test_offline_spatial_reader_requires_native_model_without_guessing_geometry():
    dataset = SimpleNamespace(
        sim_cfg=SimulationConfigs.from_dict({}, strict=True),
        spatial_enabled=True,
    )
    with pytest.raises(ValueError, match="resolved microphone geometry"):
        validate_spatial_model_geometry(_MonoModel(), dataset)
    with pytest.raises(ValueError, match="resolved microphone geometry"):
        validate_spatial_model_geometry(_SpatialModel(), dataset)
    from core.data.simulation.array_profiles import resolve_microphone_array
    dataset.resolved_microphone_array = resolve_microphone_array(
        array_profile="linear_2ch_80mm", reference_channel=1,
    )
    validate_spatial_model_geometry(_SpatialModel(), dataset)


def test_real_se_entry_injects_geometry_before_constructor():
    from tasks.se.trainer.entry import SETrainingEntry
    cfg = {"model": {"type": "SpatialNet_SE", "args": {
        "n_fft": 64, "num_layers": 1, "dim_hidden": 8, "dim_ffn": 16,
        "dim_squeeze": 4, "num_heads": 2, "conv_groups": (1, 1),
    }}}
    model = object.__new__(SETrainingEntry).build_model_for_training(cfg, 1, train_dataset=_dataset())
    assert model.n_channel == 2 and model.ref_channel == 1
    assert cfg["model"]["args"]["ref_channel"] == 1
    assert cfg["model"]["args"]["n_channel"] == 2


def test_reference_conflict_is_rejected_before_model_is_built():
    from tasks.se.trainer.entry import SETrainingEntry
    cfg = {"model": {"type": "SpatialNet_SE", "args": {"ref_channel": 0}}}
    with pytest.raises(ValueError, match="ref_channel.*Dataset geometry"):
        object.__new__(SETrainingEntry).build_model_for_training(cfg, 1, train_dataset=_dataset())


def test_offline_model_reference_is_checked_against_resolved_array():
    from core.data.simulation.array_profiles import resolve_microphone_array
    dataset = SimpleNamespace(spatial_enabled=True, resolved_microphone_array=resolve_microphone_array(
        array_profile="linear_2ch_80mm", reference_channel=1,
    ))
    with pytest.raises(ValueError, match="ref_channel"):
        validate_spatial_model_geometry(_SpatialModel(ref_channel=0), dataset)


def test_loader_startup_rejects_train_validation_reference_conflict():
    from tasks.se.trainer.entry import SETrainingEntry
    with pytest.raises(ValueError, match="train/validation microphone contract"):
        object.__new__(SETrainingEntry)._build_loaders_common(
            {}, 0, 0, 1, train_ds=_dataset(0), val_ds=_dataset(1), train_collate_fn=None,
        )


def test_offline_single_channel_does_not_require_native_multichannel_model():
    from core.data.simulation.array_profiles import resolve_microphone_array
    dataset = SimpleNamespace(spatial_enabled=True, resolved_microphone_array=resolve_microphone_array(
        custom_geometry={"positions_m": [[0., 0., 0.]]}, reference_channel=0,
    ))
    validate_spatial_model_geometry(_MonoModel(), dataset)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_SpatialModel(n_channel=4), "n_channel=4"),
        (_SpatialModel(ref_channel=0), "ref_channel=0"),
    ],
)
def test_spatial_model_geometry_rejects_model_mismatch(model, message):
    with pytest.raises(ValueError, match=message):
        validate_spatial_model_geometry(model, _dataset())
