"""SSL scene context 与 speech resource adapter 的边界合同。"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest
import torch

import core.trainer.entry as trainer_entry_module
from core.data.sampler import GroupedBatchSampler
from core.data.simulation.scene_context import (
    DatasetAcousticRuntimeAdapter,
    SceneRenderContext,
)
from core.data.speech_resource_context import DynamicSpeechResourceContext
from core.data.simulation.speech_preparation import _layer_a_config
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view
from core.trainer.entry import UnifiedTrainingEntry
from tasks.ssl.data.dynamic_dataset import SSLDynamicMixingDataset
from tasks.ssl.data.resource_adapter import (
    SSLSpeechResourceAdapter,
    _SSLSpeechPreparationDataset,
)


pytestmark = pytest.mark.unit


def _resource_context() -> DynamicSpeechResourceContext:
    """构造不触碰音频资产的最小共享资源状态。"""

    resources = object.__new__(DynamicSpeechResourceContext)
    resources.sim_cfg = SimpleNamespace(rir_backend="online")
    resources._simulation_view = object()
    resources._spatial_rir_provider = object()
    resources.speech_flt = {"speech": "speech.wav"}
    resources.speech_uids = {16000: ["speech"]}
    resources.noise_flt = {}
    resources.wn_flt = {}
    resources._worker_base_seed = torch.tensor([17], dtype=torch.int64)
    resources._worker_epoch = torch.tensor([3], dtype=torch.int64)
    resources.strict_load = True
    return resources


def test_dataset_runtime_adapter_uses_only_the_explicit_owner():
    resource_context = _resource_context()
    outer_dataset = SimpleNamespace(
        _resource_context=resource_context,
        sim_cfg=SimpleNamespace(owner="outer"),
        _simulation_view=SimpleNamespace(owner="outer"),
    )

    runtime = DatasetAcousticRuntimeAdapter(outer_dataset)

    assert runtime.config is outer_dataset.sim_cfg
    assert runtime.simulation_view is outer_dataset._simulation_view


def test_ssl_resource_context_is_explicit_and_worker_pickleable():
    resources = _resource_context()
    adapter = SSLSpeechResourceAdapter(resources, simulation_split="validation")

    assert isinstance(adapter.scene_context, SceneRenderContext)
    assert adapter.scene_context.config is resources.sim_cfg
    assert adapter.scene_context.simulation_view is resources._simulation_view
    assert adapter.speech_files is resources.speech_flt
    assert adapter.speech_uid_groups is resources.speech_uids
    assert adapter.spatial_rir_provider is resources._spatial_rir_provider
    assert adapter._worker_base_seed is resources._worker_base_seed
    assert adapter._worker_epoch is resources._worker_epoch
    assert adapter._simulation_split == "validation"

    restored = pickle.loads(pickle.dumps(adapter))
    assert restored.scene_context.config.rir_backend == "online"
    assert restored._worker_base_seed.tolist() == [17]
    assert restored._worker_epoch.tolist() == [3]


def test_ssl_layer_a_adapter_keeps_highpass_cache_on_task_owner():
    resources = _resource_context()
    task = SimpleNamespace(_hp_taps=None)
    preparation = _SSLSpeechPreparationDataset(task, resources)
    taps = {16000: torch.ones(3)}

    preparation._hp_taps = taps

    assert task._hp_taps is taps
    assert "_hp_taps" not in resources.__dict__
    assert "__getattr__" not in vars(SSLDynamicMixingDataset)


def test_ssl_layer_a_preparation_reads_the_typed_view():
    resources = _resource_context()
    config = SimulationConfigs.from_dict(
        {
            "rir_backend": "online",
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
        }
    )
    view = project_simulation_view(config, "ssl")
    class _RejectConfigReads:
        def __getattr__(self, name):
            raise AssertionError(
                f"Layer A must read the typed view, not raw config field {name!r}"
            )

    resources.sim_cfg = _RejectConfigReads()
    resources._simulation_view = view
    adapter = SSLSpeechResourceAdapter(resources, simulation_split="train")
    preparation = _SSLSpeechPreparationDataset(SimpleNamespace(), resources)

    assert _layer_a_config(preparation) is view.shared.layer_a
    assert adapter.scene_context.simulation_view is view


def test_ssl_dataset_exposes_strict_loading_preflight():
    dataset = object.__new__(SSLDynamicMixingDataset)
    dataset._resources = SimpleNamespace(strict_load=True)

    assert dataset.strict_load is True


def test_ssl_dataset_exposes_grouped_sampler_contract_explicitly():
    class _ResourceAdapter:
        balanced_speech_sampling_enabled = False

        def __len__(self):
            return 2

        def get_srs(self):
            return [16000, 16000]

        def get_source_length(self):
            return [32000, 24000]

        def get_speech_epoch_plan(self, **kwargs):
            return plan, kwargs

    plan = object()
    resource_adapter = _ResourceAdapter()
    dataset = object.__new__(SSLDynamicMixingDataset)
    dataset._resources = resource_adapter

    assert dataset.balanced_speech_sampling_enabled is False
    assert dataset.get_srs() == [16000, 16000]
    assert dataset.get_source_length() == [32000, 24000]
    sampler = GroupedBatchSampler(dataset, batch_size=2)
    assert sampler._idx_sr == [16000, 16000]
    assert sampler._full_source_length == [32000, 24000]
    result, request = dataset.get_speech_epoch_plan(
        requested=2,
        world_size=1,
        batch_size=2,
        drop_last=False,
        seed=7,
        epoch=3,
    )
    assert result is plan
    assert request == {
        "requested": 2,
        "world_size": 1,
        "batch_size": 2,
        "drop_last": False,
        "seed": 7,
        "epoch": 3,
    }


def test_ssl_dataset_preserves_statistics_epoch_and_worker_cache_hooks():
    calls = []
    summary = object()
    resource_adapter = SimpleNamespace(
        strict_load=True,
        get_data_summary=lambda *args: (calls.append(("summary", args)), summary)[1],
        set_epoch=lambda epoch, base_seed: calls.append(
            ("epoch", epoch, base_seed)
        ),
        worker_maybe_clear_caches=lambda: calls.append(("worker_cache",)),
        simulation_split="train",
        set_simulation_split=lambda value: calls.append(("split", value)),
    )
    dataset = object.__new__(SSLDynamicMixingDataset)
    dataset._resources = resource_adapter
    dataset._hp_taps = {16000: torch.ones(3)}

    assert dataset.get_data_summary("train", 5, {"num_workers": 1}) is summary
    dataset._simulation_split = "validation"
    dataset.set_epoch(4, base_seed=23)
    assert dataset._hp_taps is None
    dataset._worker_maybe_clear_caches()

    assert calls == [
        ("summary", ("train", 5, {"num_workers": 1})),
        ("split", "validation"),
        ("epoch", 4, 23),
        ("worker_cache",),
    ]


class _StopAfterLoaderPreflight(Exception):
    pass


class _SSLPreflightEntry(UnifiedTrainingEntry):
    def __init__(self, dataset):
        super().__init__("ssl-preflight-contract")
        self._dataset = dataset

    def build_loaders(self, cfg, rank, local_rank, world_size):
        return [None], [None], None, self._dataset

    def build_model(self, cfg, rank):
        raise AssertionError("loader preflight test must stop before model build")

    def build_model_for_training(self, cfg, train_ds, rank):
        raise AssertionError("loader preflight test must stop before model build")

    def get_trainer_cls(self):
        raise AssertionError("loader preflight test must stop before trainer build")


@pytest.mark.parametrize(
    ("strict_load", "expected_exception"),
    [
        (True, _StopAfterLoaderPreflight),
        (False, ValueError),
    ],
)
def test_ssl_dataset_passes_real_unified_loader_preflight(
    monkeypatch, strict_load, expected_exception
):
    dataset = object.__new__(SSLDynamicMixingDataset)
    dataset._resources = SimpleNamespace(strict_load=strict_load)
    entry = _SSLPreflightEntry(dataset)
    monkeypatch.setattr(entry, "_ensure_free_master_port", lambda: None)
    monkeypatch.setattr(
        entry,
        "load_and_setup",
        lambda **_kwargs: (SimpleNamespace(), {}, 0, 0, 1),
    )
    monkeypatch.setattr(entry, "_log_task_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        entry,
        "_log_exception_rank_aware",
        lambda exc, *_args, **_kwargs: (_ for _ in ()).throw(exc),
    )
    monkeypatch.setattr(
        trainer_entry_module,
        "detect_task",
        lambda _cfg: SimpleNamespace(
            task="ssl", subtask="contract", paradigm="dynamic", model_type="fixture"
        ),
    )
    monkeypatch.setattr(
        trainer_entry_module,
        "_ensure_file_descriptor_limit",
        lambda **_kwargs: None,
    )

    def _run_stage(stage, operation, **_kwargs):
        if stage == "loader startup preflight":
            return operation()
        raise _StopAfterLoaderPreflight()

    monkeypatch.setattr(entry, "_run_distributed_stage", _run_stage)

    with pytest.raises(expected_exception):
        entry.run(config={})
