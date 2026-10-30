"""Regression coverage for core data collation and dataset-build audit fixes."""

from __future__ import annotations

import pytest
import torch

from core.data.collate import collate_fn
from core.data.dataset_build import DatasetBuildSpec, build_dataset
from core.trainer.entry import detect_task


class _StrictDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StrictOfflineFactory:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return _StrictDataset(**request.section)


def _sample(sr=16000, length=4, physical_length=4):
    waveform = torch.arange(physical_length, dtype=torch.float32)
    return waveform, waveform.clone(), sr, length


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sr", 16000.5),
        ("sr", True),
        ("sr", "16000"),
        ("length", 2.5),
    ],
)
def test_collate_rejects_lossy_scalar_values(field, value):
    sample = list(_sample())
    sample[2 if field == "sr" else 3] = value

    with pytest.raises(TypeError, match=field):
        collate_fn([tuple(sample)])


def test_collate_accepts_one_element_integer_tensor_scalar():
    out = collate_fn([_sample(sr=torch.tensor([16000]), length=torch.tensor([4]))])

    assert out.sr.dtype is torch.int32
    assert out.sr.item() == 16000
    assert out.lengths.tolist() == [4]


def test_collate_rejects_distinct_fractional_sample_rates_before_comparison():
    with pytest.raises(TypeError, match="sr"):
        collate_fn([_sample(sr=16000.5), _sample(sr=16001.5)])


def test_collate_padding_is_bounded_by_declared_lengths():
    out = collate_fn(
        [
            _sample(length=3, physical_length=10000),
            _sample(length=2, physical_length=5),
        ]
    )

    assert out.clean.shape == (2, 3)
    assert out.noisy.shape == (2, 3)
    assert out.clean[0].tolist() == [0.0, 1.0, 2.0]
    assert out.clean[1].tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize("mode", [None, 1, "streaming"])
def test_dataset_builder_requires_explicit_known_data_mode(mode):
    cfg = {"data": {"mode": mode, "train": {"speech_scp": "speech.scp", "noise_scp": "noise.scp", "simulation_cfg": {}}}}

    with pytest.raises(ValueError, match=r"Unsupported data\.mode"):
        build_dataset(
            cfg,
            spec=DatasetBuildSpec(task="se", dynamic_dataset_cls=_StrictDataset),
        )


@pytest.mark.parametrize("mode", [None, "streaming"])
def test_detect_task_requires_explicit_known_data_mode(mode):
    with pytest.raises(ValueError, match=r"Unsupported data\.mode"):
        detect_task({"data": {"mode": mode}})


def test_offline_train_does_not_fall_back_to_validation():
    cfg = {
        "data": {
            "mode": "offline",
            "validation": {"clean_scp": "clean.scp", "noisy_scp": "noisy.scp"},
        }
    }

    with pytest.raises(ValueError, match=r"Missing data\.train"):
        build_dataset(
            cfg,
            spec=DatasetBuildSpec(task="se", dynamic_dataset_cls=_StrictDataset),
        )


@pytest.mark.parametrize("task,mode", [("se", "dynamic"), ("aec", "dynamic_aec"), ("tse", "dynamic_tse")])
def test_dynamic_build_strips_all_builder_only_metadata(task, mode):
    cfg = {
        "data": {
            "mode": mode,
            "train": {
                "speech_scp": "speech.scp",
                "noise_scp": "noise.scp",
                "simulation_cfg": {},
                "collate_fn": "unused",
                "inherit_train_simulation": True,
                "mode": "dynamic",
                "num_per_epoch": 1,
                "return_noise": True,
                "strict_load": True,
                "use_dynamic": True,
            },
        }
    }
    spec = DatasetBuildSpec(task=task, dynamic_dataset_cls=_StrictDataset)

    dataset = build_dataset(cfg, spec=spec)

    assert not set(dataset.kwargs).intersection(
        {"collate_fn", "inherit_train_simulation", "mode", "num_per_epoch", "return_noise", "use_dynamic"}
    )


@pytest.mark.parametrize("task,mode", [("se", "dynamic"), ("aec", "dynamic_aec"), ("tse", "dynamic_tse")])
def test_dynamic_validation_strips_all_builder_only_metadata(task, mode):
    speech_scp = "speech.jsonl" if task == "tse" else "speech.scp"
    validation_speech_scp = "val.jsonl" if task == "tse" else "val.scp"
    simulation_cfg = {}
    cfg = {
        "data": {
            "mode": mode,
            "train": {
                "speech_scp": speech_scp,
                "noise_scp": "noise.scp",
                **({"spk2enroll_json": "enroll.json"} if task == "tse" else {}),
                "simulation_cfg": simulation_cfg,
            },
            "validation": {
                "speech_scp": validation_speech_scp,
                "collate_fn": "unused",
                "inherit_train_simulation": True,
                "mode": "dynamic",
                "num_per_epoch": 1,
                "return_noise": True,
                "strict_load": True,
                "use_dynamic": True,
            },
        }
    }
    spec = DatasetBuildSpec(task=task, dynamic_dataset_cls=_StrictDataset)

    dataset = build_dataset(cfg, mode="val", spec=spec)

    assert not set(dataset.kwargs).intersection(
        {"collate_fn", "inherit_train_simulation", "mode", "num_per_epoch", "return_noise", "use_dynamic"}
    )


@pytest.mark.parametrize("task,mode", [("se", "offline"), ("aec", "offline_aec"), ("tse", "offline_tse")])
def test_offline_train_strips_all_builder_only_metadata(task, mode):
    factory = _StrictOfflineFactory()
    cfg = {
        "data": {
            "mode": mode,
            "train": {
                "manifest": "train.manifest",
                "collate_fn": "unused",
                "inherit_train_simulation": True,
                "mode": "offline",
                "num_per_epoch": 1,
                "return_noise": True,
                "strict_load": True,
                "use_dynamic": False,
            },
        }
    }
    spec = DatasetBuildSpec(
        task=task,
        dynamic_dataset_cls=_StrictDataset,
        offline_dataset_factory=factory,
    )

    build_dataset(cfg, spec=spec)

    assert not set(factory.requests[0].section).intersection(
        {"collate_fn", "inherit_train_simulation", "mode", "num_per_epoch", "return_noise", "strict_load", "use_dynamic"}
    )


@pytest.mark.parametrize("task,mode", [("se", "offline"), ("aec", "offline_aec"), ("tse", "offline_tse")])
def test_offline_validation_strips_all_builder_only_metadata(task, mode):
    factory = _StrictOfflineFactory()
    cfg = {
        "data": {
            "mode": mode,
            "train": {"manifest": "train.manifest"},
            "validation": {
                "manifest": "val.manifest",
                "collate_fn": "unused",
                "inherit_train_simulation": False,
                "mode": "offline",
                "num_per_epoch": 1,
                "return_noise": True,
                "strict_load": True,
                "use_dynamic": False,
            },
        }
    }
    spec = DatasetBuildSpec(
        task=task,
        dynamic_dataset_cls=_StrictDataset,
        offline_dataset_factory=factory,
    )

    build_dataset(cfg, mode="val", spec=spec)

    assert not set(factory.requests[0].section).intersection(
        {"collate_fn", "inherit_train_simulation", "mode", "num_per_epoch", "return_noise", "strict_load", "use_dynamic"}
    )


pytestmark = [pytest.mark.unit]
