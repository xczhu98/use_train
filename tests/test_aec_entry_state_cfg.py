"""Tests for AECTrainingEntry state-detector config derivation."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tasks.aec.trainer.entry import AECTrainingEntry
from core.utils.config import load_merged_config
from tasks.se.trainer.entry import SETrainingEntry
from tasks.tse.trainer.entry import TSETrainingEntry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AEC_CONFIG_ROOT = PROJECT_ROOT / "configs" / "aec"
AEC_CONFIG_PATHS = tuple(
    sorted((*AEC_CONFIG_ROOT.rglob("*.yaml"), *AEC_CONFIG_ROOT.rglob("*.yml")))
)
DBGNET_HYBRID_CONFIG_PATHS = (
    AEC_CONFIG_ROOT
    / "microsoft_challenge"
    / "dbgnet"
    / "dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml",
    AEC_CONFIG_ROOT
    / "microsoft_challenge"
    / "dbgnet"
    / "dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml",
)


def _base_cfg(label_type="fst_dt_nst", has_state_ce=True, explicit_nclass=None, explicit_use=None):
    components = [{"type": "SISNRLoss", "weight": 1.0}]
    if has_state_ce:
        components.append({"type": "StateCrossEntropyLoss", "weight": 0.001})
    cfg = {
        "data": {"train": {"state_detector_label_type": label_type}},
        "loss": {"type": "CompositeLoss", "components": components},
        "model": {"type": "DBGNet", "args": {"sample_rate": 16000}},
    }
    if explicit_nclass is not None:
        cfg["model"]["args"]["state_nclass"] = explicit_nclass
    if explicit_use is not None:
        cfg["model"]["args"]["use_state_detector"] = explicit_use
    return cfg


def test_derives_state_nclass_from_label_type():
    cfg = _base_cfg(label_type="fst_dt_nst")
    AECTrainingEntry._resolve_state_detector_cfg(cfg)
    assert cfg["model"]["args"]["state_nclass"] == 3

    cfg = _base_cfg(label_type="scenario")
    AECTrainingEntry._resolve_state_detector_cfg(cfg)
    assert cfg["model"]["args"]["state_nclass"] == 8


def test_derives_use_state_detector_from_loss():
    cfg = _base_cfg(has_state_ce=True)
    AECTrainingEntry._resolve_state_detector_cfg(cfg)
    assert cfg["model"]["args"]["use_state_detector"] is True

    cfg = _base_cfg(has_state_ce=False)
    AECTrainingEntry._resolve_state_detector_cfg(cfg)
    assert cfg["model"]["args"]["use_state_detector"] is False
    assert "state_detector_label_type" not in cfg["data"]["train"]


def test_validates_state_nclass_mismatch():
    cfg = _base_cfg(label_type="scenario", explicit_nclass=3)
    with pytest.raises(ValueError, match="state_nclass=3 does not match"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)


def test_validates_use_state_detector_mismatch():
    cfg = _base_cfg(has_state_ce=True, explicit_use=False)
    with pytest.raises(ValueError, match="use_state_detector=False is inconsistent"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)


def test_raises_when_state_ce_without_label_type():
    cfg = _base_cfg(has_state_ce=True)
    del cfg["data"]["train"]["state_detector_label_type"]
    with pytest.raises(ValueError, match="state_detector_label_type is not set"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)


def test_unknown_label_type_raises():
    cfg = _base_cfg(label_type="unknown")
    with pytest.raises(ValueError, match="Unknown state_detector_label_type"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)


@pytest.mark.parametrize("uses_generator", [False, True])
def test_dbgnet_resolution_is_idempotent(uses_generator):
    cfg = _base_cfg(label_type="scenario")
    if uses_generator:
        cfg["model"] = {"generator": cfg["model"]}
    target_cfg = cfg["model"]["generator"] if uses_generator else cfg["model"]

    AECTrainingEntry._resolve_state_detector_cfg(cfg)
    first_args = dict(target_cfg["args"])
    AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert target_cfg["args"] == first_args
    assert target_cfg["args"]["state_nclass"] == 8
    assert target_cfg["args"]["use_state_detector"] is True


@pytest.mark.parametrize("uses_generator", [False, True])
def test_path_only_dbgnet_resolution_is_idempotent(uses_generator):
    cfg = _base_cfg(label_type="scenario")
    cfg["model"].pop("type")
    cfg["model"]["path"] = "tasks.aec.models.discriminative.dbgnet.DBGNet"
    if uses_generator:
        cfg["model"] = {"generator": cfg["model"]}
    target_cfg = cfg["model"]["generator"] if uses_generator else cfg["model"]

    AECTrainingEntry._resolve_state_detector_cfg(cfg)
    first_args = dict(target_cfg["args"])
    AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert target_cfg["args"] == first_args
    assert target_cfg["args"]["state_nclass"] == 8
    assert target_cfg["args"]["use_state_detector"] is True


def test_path_takes_precedence_over_conflicting_type():
    cfg = _base_cfg()
    cfg["model"].update(
        {
            "type": "DBGNet",
            "path": "tasks.aec.models.discriminative.deepvqe.DeepVQE",
        }
    )
    original = deepcopy(cfg)

    with pytest.raises(ValueError, match="StateCrossEntropyLoss.*DeepVQE.*not DBGNet"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert cfg == original


@pytest.mark.parametrize(
    "model_cfg",
    [
        {"type": "DeepVQE", "args": {"variant": "small"}},
        {
            "generator": {
                "type": "DeepVQE",
                "args": {"variant": "small"},
            }
        },
    ],
)
def test_non_dbgnet_does_not_receive_state_detector_args(model_cfg):
    cfg = {
        "data": {"train": {"state_detector_label_type": "fst_dt_nst"}},
        "loss": {"type": "CompositeLoss", "components": []},
        "model": model_cfg,
    }

    AECTrainingEntry._resolve_state_detector_cfg(cfg)

    target_args = (
        cfg["model"]["generator"]["args"]
        if "generator" in cfg["model"]
        else cfg["model"]["args"]
    )
    assert "state_nclass" not in target_args
    assert "use_state_detector" not in target_args
    assert "state_detector_label_type" not in cfg["data"]["train"]


@pytest.mark.parametrize(
    "model_cfg",
    [
        {"type": "DeepVQE", "args": {"variant": "small"}},
        {
            "generator": {
                "type": "DeepVQE",
                "args": {"variant": "small"},
            }
        },
    ],
)
def test_non_dbgnet_rejects_state_ce(model_cfg):
    cfg = _base_cfg()
    cfg["model"] = model_cfg

    with pytest.raises(ValueError, match="StateCrossEntropyLoss.*DBGNet"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)


@pytest.mark.parametrize("weight", [0, 0.0, "0", "0.0"])
def test_zero_weight_state_ce_is_inactive(weight):
    cfg = _base_cfg()
    cfg["loss"]["components"][1]["weight"] = weight

    AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert "state_nclass" not in cfg["model"]["args"]
    assert cfg["model"]["args"]["use_state_detector"] is False
    assert "state_detector_label_type" not in cfg["data"]["train"]


def test_inactive_state_ce_removes_explicit_label_framing_from_all_splits():
    cfg = _base_cfg(has_state_ce=False)
    cfg["data"]["train"].update(
        {
            "state_detector_hop_length": 160,
            "state_detector_win_length": 320,
        }
    )
    cfg["data"]["validation"] = {
        "state_detector_label_type": "scenario",
        "state_detector_hop_length": 80,
        "state_detector_win_length": 160,
    }

    AECTrainingEntry._resolve_state_detector_cfg(cfg)

    for section_name in ("train", "validation"):
        section = cfg["data"][section_name]
        assert "state_detector_label_type" not in section
        assert "state_detector_hop_length" not in section
        assert "state_detector_win_length" not in section


def test_auxiliary_only_state_ce_is_rejected_without_mutating_model_args():
    cfg = _base_cfg()
    cfg["loss"]["components"] = [
        {"type": "StateCrossEntropyLoss", "weight": 0.001}
    ]
    original = deepcopy(cfg)

    with pytest.raises(ValueError, match="auxiliary-only"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert cfg == original


@pytest.mark.parametrize("weight", [0.001, "0.001"])
def test_nonzero_weight_state_ce_is_active(weight):
    cfg = _base_cfg()
    cfg["loss"]["components"][1]["weight"] = weight

    AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert cfg["model"]["args"]["use_state_detector"] is True


def test_state_ce_without_weight_is_active():
    cfg = _base_cfg()
    cfg["loss"]["components"][1].pop("weight")

    AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert cfg["model"]["args"]["use_state_detector"] is True


def test_top_level_state_ce_requires_composite_component():
    cfg = _base_cfg()
    cfg["loss"] = {"type": "StateCrossEntropyLoss"}
    original = deepcopy(cfg)

    with pytest.raises(ValueError, match="must be a CompositeLoss component"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert cfg == original


def test_failed_dbgnet_validation_does_not_partially_inject_args():
    cfg = _base_cfg(label_type="scenario", explicit_use=False)
    original = deepcopy(cfg)

    with pytest.raises(ValueError, match="use_state_detector=False is inconsistent"):
        AECTrainingEntry._resolve_state_detector_cfg(cfg)

    assert cfg == original


@pytest.mark.parametrize(
    "config_path",
    DBGNET_HYBRID_CONFIG_PATHS,
    ids=lambda path: path.stem,
)
def test_explicit_statece_entry_matches_direct_build_structure(config_path):
    import tasks.aec.models
    from core.models.build import build_model

    base_cfg = load_merged_config(config_path)
    assert base_cfg["model"]["args"]["use_state_detector"] is False
    assert base_cfg["loss"]["type"] == "CompositeLoss"
    assert [component["type"] for component in base_cfg["loss"]["components"]] == [
        "GTCRNCompressedSpectralLoss",
        "GTCRNLogRatioLoss",
    ]
    assert [component["weight"] for component in base_cfg["loss"]["components"]] == [
        1.0,
        1.0,
    ]

    cfg = deepcopy(base_cfg)
    cfg["model"]["args"]["use_state_detector"] = True
    cfg["loss"]["components"].append(
        {
            "type": "StateCrossEntropyLoss",
            "weight": 0.001,
            "name": "frame-level state CE engineering mode",
        }
    )
    model_args = cfg["model"]["args"]
    stft_cfg = cfg["stft"]
    assert model_args["state_nclass"] == 3
    assert [component["type"] for component in cfg["loss"]["components"]] == [
        "GTCRNCompressedSpectralLoss",
        "GTCRNLogRatioLoss",
        "StateCrossEntropyLoss",
    ]
    assert [component["weight"] for component in cfg["loss"]["components"]] == [
        1.0,
        1.0,
        0.001,
    ]
    assert cfg["loss"]["components"][2]["name"] == (
        "frame-level state CE engineering mode"
    )

    tasks.aec.models.register_aec_models(only="DBGNet")
    direct_model = build_model(deepcopy(cfg["model"]), stft_cfg=stft_cfg)
    entry_cfg = deepcopy(cfg)
    entry_model = object.__new__(AECTrainingEntry).build_model(entry_cfg, rank=1)

    direct_structure = {
        name: tuple(value.shape) for name, value in direct_model.state_dict().items()
    }
    entry_structure = {
        name: tuple(value.shape) for name, value in entry_model.state_dict().items()
    }
    assert entry_structure == direct_structure
    assert entry_cfg["model"] == cfg["model"]
    for model in (direct_model, entry_model):
        assert model.sample_rate == cfg["data"]["sample_rate"]
        assert model.n_fft == stft_cfg["n_fft"]
        assert model.hop_length == stft_cfg["hop_length"]
        assert model.win_length == stft_cfg["win_length"]
        assert model.erb.low_bins == model_args["erb_low_bins"]
        assert model.erb.erb_bands == model_args["erb_bands"]


@pytest.mark.parametrize("config_path", AEC_CONFIG_PATHS, ids=lambda path: str(path.relative_to(PROJECT_ROOT)))
def test_all_aec_yaml_build_model_smoke(config_path):
    cfg = load_merged_config(config_path)
    entry = object.__new__(AECTrainingEntry)

    model = entry.build_model(cfg, rank=1)

    assert model is not None


def _capture_loader_num_per_epoch(monkeypatch, data_sections, *, expect_state=False):
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
    components = (
        [
            {"type": "SISNRLoss", "weight": 1.0},
            {"type": "StateCrossEntropyLoss", "weight": 0.001},
        ]
        if expect_state
        else []
    )
    cfg = {
        "data": {
            "mode": "dynamic_aec",
            "sample_rate": 16000,
            "placeholder_num_samples": 1024,
            "aec_collate_raise_on_all_failed": False,
            "dataloader": {"batch_size": 1, "num_workers": 0},
            **data_sections,
        },
        "meta": {"seed": 0},
        "loss": {"type": "CompositeLoss", "components": components},
        "model": {"type": "DBGNet", "args": {"sample_rate": 16000}},
    }
    train_ds = type("Dataset", (), {})()
    val_ds = type("Dataset", (), {"set_epoch": lambda *_args, **_kwargs: None})()
    datasets = iter((train_ds, val_ds))
    monkeypatch.setattr(
        "tasks.aec.data.dynamic_aec_build.build_dynamic_aec_dataset",
        lambda *_args, **_kwargs: next(datasets),
    )
    captured = {}

    def capture_loaders(*_args, **kwargs):
        captured.update(kwargs)
        return "loaders"

    monkeypatch.setattr(entry, "_build_loaders_common", capture_loaders)

    result = entry.build_loaders(cfg, rank=0, local_rank=0, world_size=1)

    assert result == "loaders"
    return captured


@pytest.mark.parametrize("expect_state", [False, True])
def test_build_loaders_binds_all_failed_state_fallback(monkeypatch, expect_state):
    train_cfg = {"num_per_epoch": 321}
    if expect_state:
        train_cfg["state_detector_label_type"] = "fst_dt_nst"
    captured = _capture_loader_num_per_epoch(
        monkeypatch,
        {"train": train_cfg},
        expect_state=expect_state,
    )

    collate_fn = captured["train_collate_fn"]
    assert collate_fn.keywords["expect_state"] is expect_state
    batch = collate_fn([None, None])
    assert (batch.state is not None) is expect_state
    if expect_state:
        assert batch.state.dtype == torch.int64
        assert batch.state.shape == (2, 1)
        assert (batch.state == -1).all()


def test_build_loaders_forwards_train_num_per_epoch(monkeypatch):
    captured = _capture_loader_num_per_epoch(
        monkeypatch,
        {"train": {"num_per_epoch": 321}},
    )

    assert captured["train_num_per_epoch"] == 321


def test_build_loaders_forwards_validation_num_per_epoch(monkeypatch):
    captured = _capture_loader_num_per_epoch(
        monkeypatch,
        {"validation": {"num_per_epoch": 44}},
    )

    assert captured["val_num_per_epoch"] == 44


def test_build_loaders_rejects_retired_val_section():
    from core.data.dataset_build import data_plane_for_build

    with pytest.raises(ValueError, match=r"data\.val is no longer supported"):
        data_plane_for_build(
            {"data": {"mode": "dynamic_aec", "train": {}, "val": {"num_per_epoch": 55}}}
        )


class _LoaderDataset:
    def __init__(self, size=0):
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return index

    def get_srs(self):
        return [16000] * self._size

    def get_source_length(self):
        return list(range(self._size))


def _build_shared_loaders(
    entry,
    *,
    train_num_per_epoch,
    val_num_per_epoch,
    train_ds=None,
    val_ds=None,
    batch_size=1,
    rank=0,
    world_size=1,
    dataloader_cfg=None,
):
    loader_cfg = {
        "batch_size": batch_size,
        "num_workers": 0,
        "persistent_workers": False,
    }
    loader_cfg.update(dataloader_cfg or {})
    return entry._build_loaders_common(
        {
            "data": {"dataloader": loader_cfg},
            "meta": {"seed": 17},
        },
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        train_ds=train_ds or _LoaderDataset(),
        val_ds=val_ds or _LoaderDataset(),
        train_collate_fn=lambda batch: batch,
        train_num_per_epoch=train_num_per_epoch,
        val_num_per_epoch=val_num_per_epoch,
    )


def _flatten_loader_indices(loader):
    batches = list(loader)
    return batches, [index for batch in batches for index in batch]


def _entry_loader_cfg(entry_cls):
    data = {
        "mode": "dynamic",
        "sample_rate": 16000,
        "placeholder_num_samples": 1024,
        "dataloader": {
            "batch_size": 2,
            "num_workers": 0,
            "persistent_workers": False,
        },
        "train": {"num_per_epoch": 4},
        "validation": {"num_per_epoch": 3},
    }
    cfg = {"data": data, "meta": {"seed": 17}}
    if entry_cls is AECTrainingEntry:
        data["mode"] = "dynamic_aec"
        cfg.update(
            {
                "loss": {"type": "CompositeLoss", "components": []},
                "model": {"type": "DBGNet", "args": {"sample_rate": 16000}},
            }
        )
    elif entry_cls is TSETrainingEntry:
        data["mode"] = "dynamic_tse"
    return cfg


@pytest.mark.parametrize("entry_cls", [SETrainingEntry, AECTrainingEntry, TSETrainingEntry])
def test_task_entry_build_loaders_uses_real_shared_sampler_contract(monkeypatch, entry_cls):
    train_ds = _LoaderDataset(10)
    val_ds = _LoaderDataset(10)

    def build_dataset(*args, mode=None, **_kwargs):
        effective_mode = mode if mode is not None else args[1]
        return train_ds if effective_mode == "train" else val_ds

    collate = lambda batch, **_kwargs: batch
    if entry_cls is SETrainingEntry:
        monkeypatch.setattr("tasks.se.data.build_dataset", build_dataset)
        monkeypatch.setattr("tasks.se.data.collate_fn", collate)
    elif entry_cls is AECTrainingEntry:
        monkeypatch.setattr(
            "tasks.aec.data.dynamic_aec_build.build_dynamic_aec_dataset", build_dataset
        )
        monkeypatch.setattr("tasks.aec.data.collate.aec_collate_fn", collate)
    else:
        from tasks.tse.cue_contract import TSECueSpecs

        train_ds.cue_specs = TSECueSpecs()
        val_ds.cue_specs = TSECueSpecs()
        monkeypatch.setattr("tasks.tse.data.build_dataset", build_dataset)
        monkeypatch.setattr("tasks.tse.data.collate.tse_collate_fn", collate)

    entry = object.__new__(entry_cls)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
    train_loader, val_loader, _, _ = entry.build_loaders(
        _entry_loader_cfg(entry_cls), rank=0, local_rank=0, world_size=1
    )

    assert train_loader.batch_sampler.require_non_empty_all_ranks is False
    assert val_loader.batch_sampler.require_non_empty_all_ranks is True
    assert len(_flatten_loader_indices(val_loader)[1]) == 3


def test_shared_loader_validation_first_iteration_honors_virtual_epoch_cardinality():
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()

    _, val_loader, _, _ = _build_shared_loaders(
        entry,
        train_num_per_epoch=None,
        val_num_per_epoch=3,
        train_ds=_LoaderDataset(10),
        val_ds=_LoaderDataset(10),
        batch_size=2,
    )

    batches, indices = _flatten_loader_indices(val_loader)
    assert len(indices) == 3
    assert len(batches) == len(val_loader) == 2


def test_shared_loader_validation_inherits_worker_settings():
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()

    train_loader, val_loader, _, _ = _build_shared_loaders(
        entry,
        train_num_per_epoch=None,
        val_num_per_epoch=None,
        dataloader_cfg={
            "num_workers": 2,
            "persistent_workers": False,
            "prefetch_factor": 3,
        },
    )

    assert train_loader.num_workers == val_loader.num_workers == 2
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False
    assert train_loader.prefetch_factor == val_loader.prefetch_factor == 3


def test_shared_loader_validation_supports_independent_worker_settings():
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()

    train_loader, val_loader, _, _ = _build_shared_loaders(
        entry,
        train_num_per_epoch=None,
        val_num_per_epoch=None,
        dataloader_cfg={
            "num_workers": 3,
            "persistent_workers": True,
            "val_num_workers": 1,
            "val_persistent_workers": False,
        },
    )

    assert train_loader.num_workers == 3
    assert train_loader.persistent_workers is True
    assert val_loader.num_workers == 1
    assert val_loader.persistent_workers is False


def test_shared_loader_zero_workers_force_persistent_workers_off():
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()

    train_loader, val_loader, _, _ = _build_shared_loaders(
        entry,
        train_num_per_epoch=None,
        val_num_per_epoch=None,
        dataloader_cfg={
            "num_workers": 0,
            "persistent_workers": True,
            "val_num_workers": 0,
            "val_persistent_workers": True,
            "prefetch_factor": 3,
            "timeout": 120,
        },
    )

    assert train_loader.num_workers == val_loader.num_workers == 0
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False
    assert train_loader.prefetch_factor is None
    assert val_loader.prefetch_factor is None
    assert train_loader.timeout == val_loader.timeout == 0
    assert list(train_loader) == list(val_loader) == []


def test_shared_loader_worker_arguments_are_scoped_per_loader():
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()

    train_loader, val_loader, _, _ = _build_shared_loaders(
        entry,
        train_num_per_epoch=None,
        val_num_per_epoch=None,
        dataloader_cfg={
            "num_workers": 0,
            "persistent_workers": True,
            "val_num_workers": 1,
            "val_persistent_workers": True,
            "prefetch_factor": 3,
            "timeout": 120,
        },
    )

    assert train_loader.num_workers == 0
    assert train_loader.persistent_workers is False
    assert train_loader.prefetch_factor is None
    assert train_loader.timeout == 0
    assert val_loader.num_workers == 1
    assert val_loader.persistent_workers is True
    assert val_loader.prefetch_factor == 3
    assert val_loader.timeout == 120


def test_shared_loader_validation_none_keeps_full_cardinality():
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()

    _, val_loader, _, _ = _build_shared_loaders(
        entry,
        train_num_per_epoch=None,
        val_num_per_epoch=None,
        train_ds=_LoaderDataset(10),
        val_ds=_LoaderDataset(10),
        batch_size=2,
    )

    batches, indices = _flatten_loader_indices(val_loader)
    assert len(indices) == 10
    assert len(batches) == len(val_loader) == 5


def test_shared_loader_validation_epoch_zero_is_deterministic():
    def build_validation_indices():
        entry = object.__new__(AECTrainingEntry)
        entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
        _, val_loader, _, _ = _build_shared_loaders(
            entry,
            train_num_per_epoch=None,
            val_num_per_epoch=3,
            train_ds=_LoaderDataset(10),
            val_ds=_LoaderDataset(10),
            batch_size=2,
        )
        return _flatten_loader_indices(val_loader)[1]

    assert build_validation_indices() == build_validation_indices()


def test_shared_loader_validation_virtual_epoch_stride_shards_share_three_indices(
    monkeypatch,
):
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *_args, **_kwargs: None)
    rank_indices = []
    for rank in range(2):
        entry = object.__new__(AECTrainingEntry)
        entry.logger = type(
            "Logger",
            (),
            {
                "info": lambda *_args, **_kwargs: None,
                "warning": lambda *_args, **_kwargs: None,
            },
        )()
        _, val_loader, _, _ = _build_shared_loaders(
            entry,
            train_num_per_epoch=None,
            val_num_per_epoch=3,
            train_ds=_LoaderDataset(10),
            val_ds=_LoaderDataset(10),
            batch_size=2,
            rank=rank,
            world_size=2,
        )
        rank_indices.append(_flatten_loader_indices(val_loader)[1])

    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert len(rank_indices[0] + rank_indices[1]) == 3


class _BalancedLoaderDataset(_LoaderDataset):
    balanced_speech_sampling_enabled = True

    def __init__(self, size=10):
        super().__init__(size)
        self.plan_requests = []
        self.plans = []

    def get_speech_epoch_plan(self, **request):
        self.plan_requests.append(request)
        actual = request["requested"]
        indices = tuple(range(actual))
        plan = {
            "selected_indices": indices,
            "selected_uids": tuple(f"uid-{index}" for index in indices),
            "speech_plan": {
                "actual": actual,
                "sr_quotas": {16000: actual},
                "cell_quotas": {("zh", "set", 16000): actual},
            },
        }
        self.plans.append(plan)
        return plan


def test_shared_loader_balanced_validation_plans_once_through_first_iteration():
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
    val_ds = _BalancedLoaderDataset()

    _, val_loader, _, _ = _build_shared_loaders(
        entry,
        train_num_per_epoch=None,
        val_num_per_epoch=3,
        train_ds=_LoaderDataset(10),
        val_ds=val_ds,
        batch_size=2,
    )
    _, consumed_indices = _flatten_loader_indices(val_loader)

    assert len(val_ds.plan_requests) == len(val_ds.plans) == 1
    assert val_ds.plan_requests[0]["epoch"] == 0
    assert tuple(consumed_indices) == val_ds.plans[0]["selected_indices"]
    assert len(consumed_indices) == len(set(consumed_indices)) == 3
    consumed_uids = [f"uid-{index}" for index in consumed_indices]
    assert tuple(consumed_uids) == val_ds.plans[0]["selected_uids"]
    assert len(consumed_uids) == len(set(consumed_uids))


def test_shared_loader_rejects_validation_virtual_epoch_smaller_than_world_size(
    monkeypatch,
):
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
    sampler_calls = []

    def fail_if_constructed(*args, **kwargs):
        sampler_calls.append((args, kwargs))
        raise AssertionError("sampler must not be constructed for invalid DDP quota")

    monkeypatch.setattr("core.data.sampler.GroupedBatchSampler", fail_if_constructed)

    with pytest.raises(ValueError, match=r"data\.validation\.num_per_epoch.*world_size=2"):
        _build_shared_loaders(
            entry,
            train_num_per_epoch=None,
            val_num_per_epoch=1,
            world_size=2,
        )

    assert sampler_calls == []


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "10", np.int64(1)])
@pytest.mark.parametrize(
    ("keyword", "field"),
    [
        ("train_num_per_epoch", "data.train.num_per_epoch"),
        ("val_num_per_epoch", "data.validation.num_per_epoch"),
    ],
)
def test_shared_loader_rejects_invalid_num_per_epoch_before_sampler_construction(
    monkeypatch, invalid, keyword, field
):
    entry = object.__new__(AECTrainingEntry)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
    sampler_calls = []

    def fail_if_constructed(*args, **kwargs):
        sampler_calls.append((args, kwargs))
        raise AssertionError("sampler must not be constructed for invalid num_per_epoch")

    monkeypatch.setattr("core.data.sampler.GroupedBatchSampler", fail_if_constructed)
    values = {"train_num_per_epoch": 7, "val_num_per_epoch": 11}
    values[keyword] = invalid

    with pytest.raises(ValueError) as exc_info:
        _build_shared_loaders(entry, **values)

    message = str(exc_info.value)
    assert field in message
    assert repr(invalid) in message
    assert sampler_calls == []


@pytest.mark.parametrize("entry_cls", [SETrainingEntry, AECTrainingEntry, TSETrainingEntry])
@pytest.mark.parametrize(
    ("train_num_per_epoch", "val_num_per_epoch"),
    [(None, None), (7, 11), (None, 11), (7, None)],
)
def test_shared_loader_scopes_non_empty_gate_to_all_validation_samplers(
    monkeypatch, entry_cls, train_num_per_epoch, val_num_per_epoch
):
    entry = object.__new__(entry_cls)
    entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
    sampler_calls = []
    set_epoch_calls = []

    class SamplerSpy:
        def __init__(self, _dataset, **kwargs):
            self.position = len(sampler_calls)
            sampler_calls.append(kwargs)
            self._local_samples = 1

        def set_epoch(self, epoch):
            set_epoch_calls.append((self.position, epoch))

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    monkeypatch.setattr("core.data.sampler.GroupedBatchSampler", SamplerSpy)

    _build_shared_loaders(
        entry,
        train_num_per_epoch=train_num_per_epoch,
        val_num_per_epoch=val_num_per_epoch,
    )

    assert [call["num_per_epoch"] for call in sampler_calls] == [
        train_num_per_epoch,
        val_num_per_epoch,
    ]
    assert sampler_calls[0]["require_non_empty_train_shards"] is True
    assert "require_non_empty_all_ranks" not in sampler_calls[0]
    assert sampler_calls[1]["require_non_empty_all_ranks"] is True
    assert set_epoch_calls == [(1, 0)]


@pytest.mark.parametrize("entry_cls", [SETrainingEntry, AECTrainingEntry, TSETrainingEntry])
@pytest.mark.parametrize("train_num_per_epoch", [None, 4, 99])
def test_shared_loader_train_full_path_keeps_dummy_padding(
    monkeypatch, entry_cls, train_num_per_epoch
):
    class MultiSrDataset(_LoaderDataset):
        def get_srs(self):
            return [8000, 8000, 8000, 16000]

    monkeypatch.setattr(
        "torch.distributed.all_reduce", lambda tensor, op=None: tensor.fill_(1)
    )
    rank_results = []
    for rank in range(2):
        entry = object.__new__(entry_cls)
        entry.logger = type(
            "Logger",
            (),
            {
                "info": lambda *_args, **_kwargs: None,
                "warning": lambda *_args, **_kwargs: None,
            },
        )()
        train_loader, _, _, _ = _build_shared_loaders(
            entry,
            train_num_per_epoch=train_num_per_epoch,
            val_num_per_epoch=None,
            train_ds=MultiSrDataset(4),
            val_ds=_LoaderDataset(4),
            batch_size=2,
            rank=rank,
            world_size=2,
        )
        train_loader.batch_sampler.set_epoch(0)
        rank_results.append(
            (len(list(train_loader)), train_loader.batch_sampler._last_iter_real_len)
        )

    assert rank_results == [(1, 1), (1, 0)]


@pytest.mark.parametrize("entry_cls", [SETrainingEntry, AECTrainingEntry, TSETrainingEntry])
def test_shared_loader_validation_full_epoch_rejects_empty_sr_shard(
    monkeypatch, entry_cls
):
    class MultiSrDataset(_LoaderDataset):
        def get_srs(self):
            return [8000, 16000]

    collective_calls = []
    monkeypatch.setattr(
        "torch.distributed.all_reduce",
        lambda *args, **kwargs: collective_calls.append(1),
    )
    errors = []
    for rank in range(2):
        entry = object.__new__(entry_cls)
        entry.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
        with pytest.raises(ValueError, match="0 real batch") as exc_info:
            _build_shared_loaders(
                entry,
                train_num_per_epoch=None,
                val_num_per_epoch=None,
                train_ds=_LoaderDataset(4),
                val_ds=MultiSrDataset(2),
                batch_size=1,
                rank=rank,
                world_size=2,
            )
        errors.append(str(exc_info.value))
    assert errors[0] == errors[1]
    assert "rank_batch_counts=[2, 0]" in errors[0]
    assert collective_calls == []


pytestmark = [pytest.mark.unit]
