"""Trainer entry 与 sampler epoch 时序的回归合同。"""

from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast
import fcntl
import os
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset


from tasks.aec.trainer.entry import AECTrainingEntry
from core.data.sampler import GroupedBatchSampler
from core.trainer.base_trainer import BaseTrainer
from core.trainer.entry import ResumePreparation, optimizer_steps_per_epoch
from core.trainer.entry_components import validate_optimizer_parameter_coverage
from core.trainer.metrics_trainer import MetricsCapableTrainer
from core.trainer.validation_metrics_config import (
    validate_trainer_config,
    validate_validation_metric_runtime_result,
)
from core.utils.safe_load import safe_torch_load


class _MultiSrDataset(Dataset):
    def __init__(self, srs=None):
        self._srs = (
            [16000] * 5 + [8000] * 5 + [48000] * 5
            if srs is None
            else list(srs)
        )

    def __len__(self):
        return len(self._srs)

    def __getitem__(self, index):
        return index

    def get_srs(self):
        return list(self._srs)

    def get_source_length(self):
        return list(range(len(self)))


def _entry_spy(entry_cls=AECTrainingEntry):
    entry = object.__new__(entry_cls)
    entry.logger = type(
        "Logger",
        (),
        {
            "info": lambda *_args, **_kwargs: None,
            "warning": lambda *_args, **_kwargs: None,
            "exception": lambda *_args, **_kwargs: None,
        },
    )()
    return entry


def _tiny_model_optimizer():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data_cfg", "expected_loss_sample_rate", "expected_trainer_sample_rate"),
    [
        ({}, None, 16000),
        ({"sample_rate": None}, None, None),
        ({"sample_rate": 32000}, 32000, 32000),
    ],
)
def test_run_training_passes_raw_sample_rate_to_loss_but_keeps_trainer_fallback(
    monkeypatch,
    data_cfg,
    expected_loss_sample_rate,
    expected_trainer_sample_rate,
) -> None:
    observed = {}

    def build_loss_spy(_cfg, *, stft_cfg=None, sample_rate=None):
        observed["loss_stft_cfg"] = stft_cfg
        observed["loss_sample_rate"] = sample_rate
        return "loss"

    class TrainerSpy:
        def __init__(self, **kwargs):
            observed["trainer_sample_rate"] = kwargs["sample_rate"]

        def train(self):
            observed["trained"] = True

    entry = _entry_spy()
    model, optimizer = _tiny_model_optimizer()
    monkeypatch.setattr("core.losses.build.build_loss", build_loss_spy)
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)
    monkeypatch.setattr(entry, "build_scheduler", lambda *_args, **_kwargs: "scheduler")
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)

    entry.run_training(
        cfg={
            "data": dict(data_cfg),
            "loss": {},
            "stft": {"n_fft": 320, "hop_length": 160},
            "trainer": {},
        },
        args=Namespace(
            resume=False,
            resume_from=None,
            wandb=False,
            checkpoint=None,
        ),
        rank=0,
        local_rank=0,
        world_size=1,
        train_loader=DataLoader([0]),
        val_loader=DataLoader([]),
        train_sampler=None,
        train_ds=None,
        model=model,
        trainer_cls=TrainerSpy,
    )

    assert observed == {
        "loss_stft_cfg": {"n_fft": 320, "hop_length": 160},
        "loss_sample_rate": expected_loss_sample_rate,
        "trainer_sample_rate": expected_trainer_sample_rate,
        "trained": True,
    }


@pytest.mark.unit
def test_validate_trainer_config_rejects_unknown_nested_keys() -> None:
    with pytest.raises(KeyError, match="typo"):
        validate_trainer_config({"trainer": {"typo": 1}})
    with pytest.raises(KeyError, match="metric_typo"):
        validate_trainer_config(
            {"trainer": {"val_metrics": {"metric_typo": True}}}
        )


@pytest.mark.unit
def test_validate_trainer_config_preserves_legacy_numeric_and_auto_worker_values() -> None:
    validate_trainer_config(
        {
            "trainer": {
                "log_step_sub_losses": 3,
                "val_metrics": {"metrics_workers": None},
            }
        }
    )


@pytest.mark.unit
def test_validate_trainer_config_accepts_tse_dual_self_estimated_objective() -> None:
    validate_trainer_config(
        {
            "data": {"mode": "offline_tse"},
            "model": {"type": "TSE_BSRNN_SPK", "args": {"joint_training": True}},
            "trainer": {
                "objectives": {
                    "dual_self_estimated": {
                        "enabled": True,
                        "original_loss_weight": 0.4,
                        "self_loss_weight": 0.6,
                    }
                }
            }
        }
    )


@pytest.mark.unit
def test_validate_trainer_config_rejects_unknown_tse_objective_key() -> None:
    with pytest.raises(KeyError, match="objective_typo"):
        validate_trainer_config(
            {
                "data": {"mode": "offline_tse"},
                "model": {"type": "TSE_BSRNN_SPK", "args": {"joint_training": True}},
                "trainer": {
                    "objectives": {
                        "dual_self_estimated": {"objective_typo": True}
                    }
                }
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("trainer", "error", "match"),
    [
        ({"log_step_sub_losses": True}, TypeError, "must be an integer"),
        ({"log_step_sub_losses": -1}, ValueError, "must be >= 0"),
        ({"adversarial_weight": float("nan")}, ValueError, "must be finite"),
        ({"curve_figsize": [8, float("inf")]}, ValueError, "two positive"),
        ({"val_metrics": None}, TypeError, "must be a mapping"),
        ({"val_metrics": {"metrics": None}}, TypeError, "list/tuple"),
        (
            {"val_metrics": {"metrics": ["SI-SDR", "TYPO"]}},
            ValueError,
            "Unknown.*TYPO",
        ),
        (
            {"val_metrics": {"metrics": ["SI-SDR", "SI-SDR"]}},
            ValueError,
            "must not contain duplicates",
        ),
        (
            {"val_metrics": {"metrics_workers": float("inf")}},
            TypeError,
            "must be an integer",
        ),
    ],
)
def test_validate_trainer_config_rejects_invalid_runtime_values(
    trainer, error, match
) -> None:
    with pytest.raises(error, match=match):
        validate_trainer_config({"trainer": trainer})


@pytest.mark.unit
def test_validate_trainer_config_accepts_erle_and_per_metric_min_coverage() -> None:
    validate_trainer_config(
        {
            "trainer": {
                "val_metrics": {
                    "enabled": True,
                    "metrics": ["SI-SDR", "ERLE"],
                    "min_coverage": {"SI-SDR": 0.95, "ERLE": 1},
                }
            }
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("min_coverage", "error", "match"),
    [
        (None, TypeError, "must be a mapping"),
        ({"PESQ": 0.5}, ValueError, "not enabled.*PESQ"),
        ({"": 0.5}, TypeError, "keys must be non-empty"),
        ({"ERLE": True}, TypeError, "ERLE must be a number"),
        ({"ERLE": float("nan")}, ValueError, "ERLE must be finite"),
        ({"ERLE": float("inf")}, ValueError, "ERLE must be finite"),
        ({"ERLE": -0.01}, ValueError, r"ERLE must be in \[0, 1\]"),
        ({"ERLE": 1.01}, ValueError, r"ERLE must be in \[0, 1\]"),
    ],
)
def test_validate_trainer_config_rejects_invalid_min_coverage(
    min_coverage, error, match
) -> None:
    with pytest.raises(error, match=match):
        validate_trainer_config(
            {
                "trainer": {
                    "val_metrics": {
                        "metrics": ["ERLE"],
                        "min_coverage": min_coverage,
                    }
                }
            }
        )


@pytest.mark.unit
def test_aec_entry_does_not_inject_undeclared_erle(monkeypatch) -> None:
    """未声明 ERLE 的 AEC val_metrics 原样进入共享校验，不追加 ERLE。"""
    observed = {}

    def capture_shared_validation(_self, cfg):
        observed["metrics"] = list(cfg["trainer"]["val_metrics"]["metrics"])
        raise RuntimeError("shared validation reached")

    monkeypatch.setattr(
        "core.trainer.entry.UnifiedTrainingEntry._validate_config",
        capture_shared_validation,
    )
    cfg = {
        "trainer": {
            "val_metrics": {
                "enabled": True,
                "metrics": ["SI-SDR", "ESTOI"],
            }
        }
    }

    with pytest.raises(RuntimeError, match="shared validation reached"):
        _entry_spy()._validate_config(cfg)

    assert observed["metrics"] == ["SI-SDR", "ESTOI"]


@pytest.mark.unit
def test_aec_entry_keeps_omitted_val_metrics_list_absent(monkeypatch) -> None:
    """省略 metrics 时保持省略，只走共享默认列表，不追加 ERLE。"""
    observed = {}

    def capture_shared_validation(_self, cfg):
        observed["val_metrics"] = dict(cfg["trainer"]["val_metrics"])
        raise RuntimeError("shared validation reached")

    monkeypatch.setattr(
        "core.trainer.entry.UnifiedTrainingEntry._validate_config",
        capture_shared_validation,
    )
    cfg = {"trainer": {"val_metrics": {"enabled": True}}}

    with pytest.raises(RuntimeError, match="shared validation reached"):
        _entry_spy()._validate_config(cfg)

    assert observed["val_metrics"] == {"enabled": True}


@pytest.mark.unit
def test_validation_metric_runtime_result_contract() -> None:
    assert validate_validation_metric_runtime_result(
        "ERLE", count=1, eligible=100, strict=False
    )
    assert not validate_validation_metric_runtime_result(
        "ERLE", count=0, eligible=100, strict=False
    )
    with pytest.raises(RuntimeError, match="strict=True.*ERLE"):
        validate_validation_metric_runtime_result(
            "ERLE", count=0, eligible=100, strict=True
        )
    with pytest.raises(RuntimeError, match="ERLE validation failed.*worker failed"):
        validate_validation_metric_runtime_result(
            "ERLE",
            count=0,
            eligible=100,
            strict=False,
            error=RuntimeError("worker failed"),
        )
    with pytest.raises(RuntimeError, match="coverage.*below"):
        validate_validation_metric_runtime_result(
            "ERLE",
            count=0,
            eligible=100,
            strict=False,
            min_coverage=0.5,
        )
    with pytest.raises(RuntimeError, match="coverage.*below"):
        validate_validation_metric_runtime_result(
            "ERLE",
            count=49,
            eligible=100,
            strict=False,
            min_coverage=0.5,
        )
    assert validate_validation_metric_runtime_result(
        "ERLE",
        count=50,
        eligible=100,
        strict=False,
        min_coverage=0.5,
    )


@pytest.mark.unit
def test_optimizer_coverage_rejects_missing_trainable_parameter() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD([model.weight], lr=0.1)

    with pytest.raises(ValueError, match="bias"):
        validate_optimizer_parameter_coverage(model, optimizer)


@pytest.mark.unit
def test_run_wraps_rank_zero_data_statistics_in_distributed_preflight(
    monkeypatch,
) -> None:
    entry = _entry_spy()
    stages: list[str] = []
    task_info = SimpleNamespace(task="aec", paradigm="discriminative")
    train_loader = DataLoader([0])
    val_loader = DataLoader([])

    monkeypatch.setattr(entry, "_ensure_free_master_port", lambda: None)
    monkeypatch.setattr(
        entry,
        "load_and_setup",
        lambda: (Namespace(), {"trainer": {}}, 0, 0, 2),
    )
    monkeypatch.setattr("core.trainer.entry.detect_task", lambda _cfg: task_info)
    monkeypatch.setattr(
        "core.trainer.entry._ensure_file_descriptor_limit", lambda **_kwargs: None
    )
    monkeypatch.setattr(entry, "_log_task_info", lambda *_args: None)
    monkeypatch.setattr(
        entry,
        "build_loaders",
        lambda *_args: (
            train_loader,
            val_loader,
            None,
            SimpleNamespace(strict_load=True),
        ),
    )
    monkeypatch.setattr(
        entry,
        "_log_data_statistics",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("stats failed")),
    )
    monkeypatch.setattr(
        entry,
        "build_model",
        lambda *_args: pytest.fail("model build must not run after preflight failure"),
    )

    def run_stage(stage, operation, **_kwargs):
        stages.append(stage)
        return operation()

    monkeypatch.setattr(entry, "_run_distributed_stage", run_stage)

    with pytest.raises(SystemExit) as exc_info:
        entry.run()

    assert exc_info.value.code == 1
    assert stages == [
        "loader startup preflight",
        "data statistics startup preflight",
    ]


def _resume_args(tmp_path, *, start_epoch, resume):
    resume_from = None
    if resume:
        resume_from = tmp_path / "model_E0007_B000000.pth"
        torch.save(
            {"epoch": start_epoch, "batch_cursor": 1, "n_steps": 0},
            resume_from,
        )
    return Namespace(
        resume=resume,
        resume_from=str(resume_from) if resume_from is not None else None,
        wandb=False,
        checkpoint=None,
    )


def _build_virtual_train_loader(*, seed, rank=0):
    entry = _entry_spy()
    dataset = _MultiSrDataset([8000, 8000, 16000, 16000])
    train_loader, _, sampler, _ = entry._build_loaders_common(
        {
            "data": {
                "dataloader": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "persistent_workers": False,
                }
            },
            "meta": {"seed": seed},
        },
        rank=rank,
        local_rank=rank,
        world_size=2,
        train_ds=dataset,
        val_ds=_MultiSrDataset([16000, 16000]),
        train_collate_fn=lambda batch: batch,
        train_num_per_epoch=2,
    )
    return entry, train_loader, sampler


def test_fresh_start_uses_safe_epoch_one_even_when_epoch_zero_is_unsafe():
    entry, train_loader, sampler = _build_virtual_train_loader(seed=1)

    prepared = entry._prepare_train_sampler_for_scheduler(
        {"trainer": {}},
        Namespace(resume=False, resume_from=None),
        sampler,
    )

    assert prepared.start_epoch == 1
    assert prepared.snapshot_path is None
    assert sampler.epoch == 1
    assert len(train_loader) == 1


def test_fresh_start_rejects_unsafe_epoch_one_symmetrically_when_epoch_zero_is_safe():
    errors = []
    for rank in range(2):
        entry, _, sampler = _build_virtual_train_loader(seed=7, rank=rank)
        with pytest.raises(
            ValueError, match=r"epoch=1.*rank_sample_counts=\[2, 0\]"
        ) as exc_info:
            entry._prepare_train_sampler_for_scheduler(
                {"trainer": {}},
                Namespace(resume=False, resume_from=None),
                sampler,
            )
        errors.append(str(exc_info.value))

    assert errors[0] == errors[1]


def test_completed_checkpoint_prepares_next_epoch_cardinality(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "batch_cursor": 0, "n_steps": 0}, checkpoint)
    dataset = _MultiSrDataset()
    sampler = GroupedBatchSampler(
        dataset,
        batch_size=3,
        seed=1,
        num_per_epoch=7,
        drop_last=True,
    )
    args = Namespace(
        resume=True,
        resume_from=str(checkpoint),
        wandb=False,
        checkpoint=None,
    )

    entry = _entry_spy()
    staging_root = tmp_path / "run"
    prepared = entry._prepare_train_sampler_for_scheduler(
        {"trainer": {"exp_path": str(staging_root)}}, args, sampler
    )

    try:
        assert prepared.start_epoch == 8
        resolved = prepared.snapshot_path
        assert resolved is not None and resolved.is_absolute()
        assert resolved != checkpoint.resolve()
        assert resolved.parent == staging_root.resolve() / "resume-staging"
        assert sampler.epoch == 8
        assert len(sampler) == 1
    finally:
        entry._cleanup_resume_checkpoint_snapshot(prepared.snapshot_path)


@pytest.mark.parametrize("resume_from", [None, "missing/model_E0007_B000001.pth"])
def test_resume_request_without_local_checkpoint_fails_fast(tmp_path, resume_from):
    args = Namespace(
        resume=True,
        resume_from=str(tmp_path / resume_from) if resume_from is not None else None,
    )
    cfg = {"trainer": {"exp_path": str(tmp_path / "empty-exp")}}

    with pytest.raises(RuntimeError, match="resume requested but local rank found no checkpoint"):
        _entry_spy()._prepare_train_sampler_for_scheduler(cfg, args, None)


def test_checkpoint_start_epoch_rejects_malformed_schema(tmp_path):
    """resume 预览复用 trainer schema：畸形 checkpoint 报契约错误而非 KeyError。"""
    entry = _entry_spy()
    cfg = {"trainer": {}}

    good = tmp_path / "good.pth"
    torch.save({"epoch": 7, "n_steps": 3, "batch_cursor": 1}, good)
    assert entry._checkpoint_start_epoch(cfg, good) == 7

    missing = tmp_path / "missing.pth"
    torch.save({"epoch": 7, "batch_cursor": 1}, missing)
    with pytest.raises(RuntimeError, match="missing required state 'n_steps'"):
        entry._checkpoint_start_epoch(cfg, missing)

    fractional = tmp_path / "fractional.pth"
    torch.save({"epoch": 1.7, "n_steps": 3}, fractional)
    with pytest.raises(TypeError, match="must be an exact int"):
        entry._checkpoint_start_epoch(cfg, fractional)


def test_explicit_missing_resume_path_does_not_fallback_to_sibling_checkpoint(tmp_path):
    exp_path = tmp_path / "exp"
    sibling_dir = exp_path / "checkpoints" / "sibling-run"
    sibling_dir.mkdir(parents=True)
    sibling_checkpoint = sibling_dir / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, sibling_checkpoint)
    explicit_missing = exp_path / "checkpoints" / "requested-run" / "latest_model.tar"
    cfg = {"trainer": {"exp_path": str(exp_path)}}
    entry = _entry_spy()

    assert entry._resume_checkpoint_path(
        cfg, Namespace(resume=True, resume_from=str(explicit_missing))
    ) is None
    with pytest.raises(RuntimeError, match="resume requested but local rank found no checkpoint"):
        entry._prepare_train_sampler_for_scheduler(
            cfg,
            Namespace(resume=True, resume_from=str(explicit_missing)),
            None,
        )
    assert entry._resume_checkpoint_path(
        cfg, Namespace(resume=True, resume_from=None)
    ) == sibling_checkpoint


def test_resume_preview_canonicalizes_relative_checkpoint_path(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "latest_model.tar"
    checkpoint.parent.mkdir()
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    monkeypatch.chdir(tmp_path)

    entry = _entry_spy()
    staging_root = tmp_path / "run"
    prepared = entry._prepare_train_sampler_for_scheduler(
        {"trainer": {"exp_path": str(staging_root)}},
        Namespace(resume=True, resume_from="checkpoints/latest_model.tar"),
        None,
    )
    resolved = prepared.snapshot_path

    try:
        assert resolved is not None and resolved.is_absolute()
        assert resolved.parent == staging_root.resolve() / "resume-staging"
    finally:
        entry._cleanup_resume_checkpoint_snapshot(prepared.snapshot_path)


def test_freeze_resume_checkpoint_uses_writable_staging_for_read_only_source(
    monkeypatch, tmp_path
):
    source_dir = tmp_path / "read-only-source"
    source_dir.mkdir()
    checkpoint = source_dir / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    staging_dir = tmp_path / "run" / "resume-staging"

    def reject_hardlink(*_args, **_kwargs):
        raise PermissionError("source filesystem is read-only")

    monkeypatch.setattr(os, "link", reject_hardlink)
    entry = _entry_spy()
    snapshot, identity = entry._freeze_resume_checkpoint(
        checkpoint, staging_dir=staging_dir
    )

    try:
        assert snapshot.parent == staging_dir.resolve()
        assert snapshot.read_bytes() == checkpoint.read_bytes()
        assert identity[0] == checkpoint.stat().st_size
    finally:
        entry._cleanup_resume_checkpoint_snapshot(snapshot)


def test_resume_checkpoint_content_identity_ignores_path_device_and_inode(tmp_path):
    first = tmp_path / "rank0" / "latest_model.tar"
    second = tmp_path / "rank1" / "different-name.pth"
    first.parent.mkdir()
    second.parent.mkdir()
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, first)
    second.write_bytes(first.read_bytes())

    first_identity = _entry_spy()._checkpoint_content_identity(first)
    second_identity = _entry_spy()._checkpoint_content_identity(second)

    assert first.stat().st_ino != second.stat().st_ino
    assert first_identity == second_identity


@pytest.mark.unit
def test_resume_preparation_normalizes_dataclass(tmp_path):
    source = tmp_path / "source" / "latest_model.tar"
    snapshot = tmp_path / "staging" / "resume.snapshot"
    existing = ResumePreparation(9, source, snapshot)

    assert ResumePreparation.normalize(existing, source_path=source) is existing
    assert ResumePreparation.normalize(ResumePreparation(1, None, None), source_path=None) == (
        ResumePreparation(1, None, None)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        (7, Path("snapshot")),
        (8, Path("source"), Path("snapshot")),
        {"start_epoch": 1},
    ],
)
def test_resume_preparation_rejects_non_dataclass(value):
    with pytest.raises(TypeError, match="must be a ResumePreparation"):
        ResumePreparation.normalize(value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "source", "error", "match"),
    [
        (ResumePreparation(True, Path("source"), Path("snapshot")), None, TypeError, "start_epoch"),
        (ResumePreparation(0, Path("source"), Path("snapshot")), None, ValueError, "positive"),
        (ResumePreparation(1, Path("other"), Path("snapshot")), Path("source"), ValueError, "source"),
        (ResumePreparation(1, None, Path("snapshot")), None, ValueError, "source.*snapshot"),
        (ResumePreparation(1, Path("source"), None), None, ValueError, "source.*snapshot"),
    ],
)
def test_resume_preparation_rejects_invalid_dataclass(value, source, error, match):
    with pytest.raises(error, match=match):
        ResumePreparation.normalize(value, **({"source_path": source} if source is not None else {}))


def test_resume_preview_freezes_checkpoint_across_atomic_replace(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    replacement = tmp_path / "replacement.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)

    prepared = _entry_spy()._prepare_train_sampler_for_scheduler(
        {"trainer": {"exp_path": str(tmp_path / "run")}},
        Namespace(resume=True, resume_from=str(checkpoint)),
        None,
    )
    resolved = prepared.snapshot_path
    torch.save({"epoch": 12, "n_steps": 100, "batch_cursor": 1}, replacement)
    os.replace(replacement, checkpoint)

    try:
        assert prepared.start_epoch == 7
        assert resolved is not None and resolved != checkpoint.resolve()
        assert safe_torch_load(resolved)["epoch"] == 7
        assert safe_torch_load(checkpoint)["epoch"] == 12
    finally:
        if resolved is not None and resolved != checkpoint.resolve():
            resolved.unlink(missing_ok=True)


def test_run_training_loads_frozen_snapshot_after_latest_atomic_replace(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "latest_model.tar"
    replacement = tmp_path / "replacement.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    entry = _entry_spy()
    original_prepare = entry._prepare_train_sampler_for_scheduler
    observed = {}

    def prepare_then_replace(cfg, args, train_sampler):
        result = original_prepare(cfg, args, train_sampler)
        torch.save({"epoch": 12, "n_steps": 100, "batch_cursor": 1}, replacement)
        os.replace(replacement, checkpoint)
        return result

    class TrainerSpy:
        def __init__(self, **kwargs):
            assert "resume_load_from" not in kwargs
            snapshot = Path(kwargs["resume_from"])
            observed["snapshot"] = snapshot
            observed["loaded_epoch"] = safe_torch_load(snapshot)["epoch"]

        def train(self):
            return None

    monkeypatch.setattr(entry, "_prepare_train_sampler_for_scheduler", prepare_then_replace)
    monkeypatch.setattr("core.losses.build.build_loss", lambda *_args, **_kwargs: "loss")
    model, optimizer = _tiny_model_optimizer()
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)
    monkeypatch.setattr(entry, "build_scheduler", lambda *_args, **_kwargs: "scheduler")
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)

    entry.run_training(
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "trainer": {"exp_path": str(tmp_path / "run")},
        },
        args=Namespace(
            resume=True,
            resume_from=str(checkpoint),
            wandb=False,
            checkpoint=None,
        ),
        rank=0,
        local_rank=0,
        world_size=1,
        train_loader=DataLoader([0]),
        val_loader=DataLoader([]),
        train_sampler=None,
        train_ds=None,
        model=model,
        trainer_cls=TrainerSpy,
    )

    assert observed["loaded_epoch"] == 7
    assert not observed["snapshot"].exists()
    assert safe_torch_load(checkpoint)["epoch"] == 12


def test_base_trainer_loads_snapshot_but_keeps_persistent_resume_run_dirs(
    monkeypatch, tmp_path
):
    exp_path = tmp_path / "exp"
    checkpoint_dir = exp_path / "checkpoints" / "persistent-run"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "latest_model.tar"
    replacement = tmp_path / "replacement.tar"
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    state = {
        "epoch": 7,
        "batch_cursor": 1,
        "n_steps": 3,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, checkpoint)
    source_resolution_calls = []
    observed = {}

    class CompatibilityResumeEntry(AECTrainingEntry):
        def _prepare_train_sampler_for_scheduler(self, cfg, args, train_sampler):
            prepared = super()._prepare_train_sampler_for_scheduler(
                cfg, args, train_sampler
            )
            replacement_state = dict(state, epoch=12)
            torch.save(replacement_state, replacement)
            os.replace(replacement, checkpoint)
            self.__dict__.pop("_resolved_resume_source_path", None)
            return prepared

    entry = _entry_spy(CompatibilityResumeEntry)
    original_resolve = entry._resume_checkpoint_path

    def resolve_once(cfg, args):
        source_resolution_calls.append(None)
        return original_resolve(cfg, args)

    class WriterStub:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class PersistentRunTrainer(BaseTrainer):
        def __new__(cls, **kwargs):
            observed["new_resume_from"] = Path(kwargs["resume_from"])
            observed["new_resume_source_from"] = Path(
                kwargs["resume_source_from"]
            )
            return super().__new__(cls)

        def __init__(
            self,
            train_loader,
            val_loader,
            sample_rate=16000,
            stft_module=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.train_loader = train_loader
            self.val_loader = val_loader

        def _save_experiment_context(self):
            return None

        def _resume_checkpoint(self):
            observed["override_resume_from"] = Path(self._resume_from)
            observed["override_resume_load_from"] = Path(self._resume_load_from)
            return super()._resume_checkpoint()

        def _after_resume_checkpoint(self, checkpoint_state):
            observed["after_resume_epoch"] = checkpoint_state["epoch"]

        def train(self):
            observed["loaded_epoch"] = self.start_epoch
            observed["resume_from"] = Path(self._resume_from)
            observed["resume_load_from"] = Path(self._resume_load_from)
            observed["ckpt_dir"] = self.ckpt_dir
            observed["log_dir"] = self.log_dir
            self._save_checkpoint(8)

    monkeypatch.setattr(entry, "_resume_checkpoint_path", resolve_once)
    monkeypatch.setattr("core.losses.build.build_loss", lambda *_args, **_kwargs: "loss")
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)
    monkeypatch.setattr(entry, "build_scheduler", lambda *_args, **_kwargs: scheduler)
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)
    tensorboard_stub = ModuleType("torch.utils.tensorboard")
    tensorboard_stub.SummaryWriter = WriterStub
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard_stub)
    monkeypatch.setattr(torch.utils, "tensorboard", tensorboard_stub, raising=False)

    cfg = {
        "data": {"sample_rate": 16000},
        "loss": {},
        "trainer": {
            "exp_path": str(exp_path),
            "validation_epoch_interval": 0,
            "save_top_k": 0,
        },
    }
    entry.run_training(
        cfg=cfg,
        args=Namespace(
            resume=True,
            resume_from=str(checkpoint),
            wandb=False,
            checkpoint=None,
        ),
        rank=0,
        local_rank=0,
        world_size=1,
        train_loader=DataLoader([0]),
        val_loader=DataLoader([]),
        train_sampler=None,
        train_ds=None,
        model=model,
        trainer_cls=PersistentRunTrainer,
    )

    snapshot = observed["new_resume_from"]
    assert observed == {
        "new_resume_from": snapshot,
        "new_resume_source_from": checkpoint.resolve(),
        "override_resume_from": snapshot,
        "override_resume_load_from": snapshot,
        "after_resume_epoch": 7,
        "loaded_epoch": 7,
        "resume_from": snapshot,
        "resume_load_from": snapshot,
        "ckpt_dir": checkpoint_dir,
        "log_dir": exp_path / "logs" / "persistent-run",
    }
    assert snapshot != checkpoint.resolve()
    assert not snapshot.exists()
    assert source_resolution_calls == [None]
    assert safe_torch_load(checkpoint)["epoch"] == 8
    assert (checkpoint_dir / "model_0008.pth").exists()
    discovered = entry._resume_checkpoint_path(
        cfg, Namespace(resume=True, resume_from=None)
    )
    assert discovered == checkpoint
    assert not list((exp_path / "resume-staging").glob("*.snapshot"))


@pytest.mark.parametrize(
    ("start_epoch", "resume"),
    [(1, False), (7, True)],
)
@pytest.mark.parametrize(
    ("drop_last", "epoch_lengths"),
    [(False, {1: 4, 7: 3}), (True, {1: 1, 7: 2})],
)
def test_scheduler_uses_exact_first_training_epoch_cardinality(
    monkeypatch, tmp_path, start_epoch, resume, drop_last, epoch_lengths
):
    dataset = _MultiSrDataset()
    sampler = GroupedBatchSampler(
        dataset,
        batch_size=3,
        seed=1,
        num_per_epoch=7,
        drop_last=drop_last,
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler)
    set_epoch_calls = []
    original_set_epoch = sampler.set_epoch

    def track_set_epoch(epoch):
        set_epoch_calls.append(epoch)
        original_set_epoch(epoch)

    sampler.set_epoch = track_set_epoch
    observed = {}

    class TrainerSpy(BaseTrainer):
        def __init__(self, **kwargs):
            self.train_loader = kwargs["train_loader"]
            self.val_loader = kwargs["val_loader"]
            self.train_sampler = kwargs["train_sampler"]
            self.train_dataset = None
            self.start_epoch = start_epoch
            self.epochs = start_epoch
            self.current_epoch = start_epoch - 1
            self.world_size = 1
            self.rank = 0
            self.config = {"meta": {"seed": 1}}
            self.models = {}
            self.writer = None
            self.wandb_run = None
            self.validation_epoch_interval = 0
            self.validation_steps_interval = 0
            self.save_top_k = 0
            self.max_steps = None
            self.n_steps = 0
            self.clear_cache_each_epoch = False
            self.lr_per_step = True
            self.save_interval = 0
            observed["resume"] = kwargs["resume"]
            observed["resume_from"] = kwargs["resume_from"]
            observed["scheduler"] = kwargs["scheduler"]
            observed["before_epoch_len"] = len(self.train_loader)

        def _train_epoch(self, epoch, step_callback=None):
            observed["epoch_len"] = len(self.train_loader)
            observed["iteration_len"] = len(list(self.train_loader))
            return 0.0

        def _save_checkpoint(self, *_args, **_kwargs):
            return None

    entry = _entry_spy()
    monkeypatch.setattr(
        "core.losses.build.build_loss", lambda *_args, **_kwargs: "loss"
    )
    model, optimizer = _tiny_model_optimizer()
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)

    def capture_scheduler(_cfg, steps_per_epoch, _optimizer, *, world_size):
        observed["scheduler_steps"] = steps_per_epoch
        observed["scheduler_world_size"] = world_size
        return "scheduler"

    monkeypatch.setattr(entry, "build_scheduler", capture_scheduler)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)
    args = _resume_args(tmp_path, start_epoch=start_epoch, resume=resume)

    entry.run_training(
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "trainer": {"exp_path": str(tmp_path / "run")},
        },
        args=args,
        rank=0,
        local_rank=0,
        world_size=1,
        train_loader=train_loader,
        val_loader=DataLoader([]),
        train_sampler=sampler,
        train_ds=dataset,
        model=model,
        trainer_cls=TrainerSpy,
    )

    expected_len = epoch_lengths[start_epoch]
    resume_from = observed.pop("resume_from")
    assert observed == {
        "scheduler_steps": expected_len,
        "scheduler_world_size": 1,
        "resume": resume,
        "scheduler": "scheduler",
        "before_epoch_len": expected_len,
        "epoch_len": expected_len,
        "iteration_len": expected_len,
    }
    if resume:
        snapshot = Path(resume_from)
        assert snapshot != Path(args.resume_from).resolve()
        assert not snapshot.exists()
        assert Path(args.resume_from).exists()
        staging_dir = tmp_path / "run" / "resume-staging"
        assert staging_dir.is_dir()
        assert not list(staging_dir.glob("*.snapshot"))
    else:
        assert resume_from is None
    assert set_epoch_calls == [start_epoch, start_epoch]


def _run_minimal_scheduler_training(
    monkeypatch,
    entry,
    *,
    world_size,
    cfg=None,
    model=None,
    optimizer=None,
    train_loader=None,
    train_sampler=None,
):
    model = model or torch.nn.Linear(1, 1, bias=False)
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=1e-3)
    train_loader = train_loader or DataLoader([0, 1, 2], batch_size=1)
    observed = {}

    class TrainerSpy:
        def __init__(self, **kwargs):
            observed["scheduler"] = kwargs["scheduler"]

        def train(self):
            observed["trained"] = True

    monkeypatch.setattr(
        "core.losses.build.build_loss", lambda *_args, **_kwargs: "loss"
    )
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)
    entry.run_training(
        cfg=cfg or {"data": {"sample_rate": 16000}, "loss": {}, "trainer": {}},
        args=Namespace(
            resume=False, resume_from=None, wandb=False, checkpoint=None
        ),
        rank=0,
        local_rank=0,
        world_size=world_size,
        train_loader=train_loader,
        val_loader=DataLoader([]),
        train_sampler=train_sampler,
        train_ds=None,
        model=model,
        trainer_cls=TrainerSpy,
    )
    return observed


def test_run_training_passes_world_size_to_scheduler_override(monkeypatch):
    calls = []

    class CustomSchedulerEntry(AECTrainingEntry):
        def build_scheduler(self, cfg, n_train_batches, optimizer, *, world_size):
            calls.append((cfg, n_train_batches, optimizer, world_size))
            return "custom-scheduler"

    entry = _entry_spy(CustomSchedulerEntry)
    observed = _run_minimal_scheduler_training(
        monkeypatch, entry, world_size=2
    )

    assert calls and calls[0][1] == 3 and calls[0][3] == 2
    assert observed == {"scheduler": "custom-scheduler", "trained": True}


def test_run_training_rejects_three_argument_scheduler_override(monkeypatch):
    class OldSignatureSchedulerEntry(AECTrainingEntry):
        def build_scheduler(self, cfg, n_train_batches, optimizer):
            return "old-signature-scheduler"

    entry = _entry_spy(OldSignatureSchedulerEntry)
    with pytest.raises(TypeError, match="world_size"):
        _run_minimal_scheduler_training(monkeypatch, entry, world_size=2)


def test_scheduler_override_receives_run_world_size(monkeypatch):
    seen_world_sizes = []

    class WorldSizeAwareSchedulerEntry(AECTrainingEntry):
        def build_scheduler(self, cfg, n_train_batches, optimizer, *, world_size):
            seen_world_sizes.append(world_size)
            return "world-size-scheduler"

    entry = _entry_spy(WorldSizeAwareSchedulerEntry)
    _run_minimal_scheduler_training(monkeypatch, entry, world_size=4)

    assert seen_world_sizes == [4]


def test_auto_horizon_planner_failure_restores_scheduler_context(monkeypatch):
    entry = _entry_spy()
    entry._scheduler_planned_optimizer_steps = 123

    def fail_planning(*_args, **_kwargs):
        raise RuntimeError("injected planning failure")

    monkeypatch.setattr(entry, "_planned_optimizer_steps", fail_planning)
    with pytest.raises(RuntimeError, match="injected planning failure"):
        _run_minimal_scheduler_training(
            monkeypatch,
            entry,
            world_size=2,
            cfg={
                "data": {"sample_rate": 16000},
                "loss": {},
                "scheduler": {
                    "type": "LinearWarmupCosineAnnealing",
                    "args": {
                        "warmup_steps": "auto",
                        "decay_until_step": 20,
                    },
                },
                "trainer": {"epochs": 3, "lr_schedule_per_step": True},
            },
        )

    assert entry._scheduler_planned_optimizer_steps == 123


def test_custom_scheduler_override_does_not_invoke_auto_horizon_planner(monkeypatch):
    class ThrowingSampler:
        def set_epoch(self, _epoch):
            pass

        def batch_count_for_epoch(self, _epoch):
            raise AssertionError("custom override must not plan all epochs")

    class CustomSchedulerEntry(AECTrainingEntry):
        def build_scheduler(self, cfg, n_train_batches, optimizer, *, world_size):
            assert world_size == 2
            return "custom-scheduler"

    entry = _entry_spy(CustomSchedulerEntry)
    observed = _run_minimal_scheduler_training(
        monkeypatch,
        entry,
        world_size=2,
        train_sampler=ThrowingSampler(),
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "args": {
                    "warmup_steps": "auto",
                    "decay_until_step": "auto",
                },
            },
            "trainer": {"epochs": 3, "lr_schedule_per_step": True},
        },
    )

    assert observed == {"scheduler": "custom-scheduler", "trained": True}


@pytest.mark.parametrize(
    "scheduler_cfg",
    [
        {
            "type": "LinearWarmupCosineAnnealing",
            "args": {"warmup_steps": 2, "decay_until_step": 20},
        },
        {"type": "StepLR", "args": {"step_size": 2}},
    ],
)
def test_base_scheduler_skips_full_plan_when_auto_horizon_is_not_consumed(
    monkeypatch, scheduler_cfg
):
    class ThrowingSampler:
        def set_epoch(self, _epoch):
            pass

        def batch_count_for_epoch(self, _epoch):
            raise AssertionError("fixed or unrelated scheduler must not plan all epochs")

    entry = _entry_spy()
    observed = _run_minimal_scheduler_training(
        monkeypatch,
        entry,
        world_size=1,
        train_sampler=ThrowingSampler(),
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "scheduler": scheduler_cfg,
            "trainer": {"epochs": 3, "lr_schedule_per_step": True},
        },
    )

    assert observed["trained"] is True


@pytest.mark.parametrize(("world_size", "expected_steps_per_epoch"), [(1, 2), (2, 1)])
def test_base_scheduler_uses_world_size_for_single_card_ceil_and_ddp_floor(
    monkeypatch, world_size, expected_steps_per_epoch
):
    entry = _entry_spy()
    observed = _run_minimal_scheduler_training(
        monkeypatch,
        entry,
        world_size=world_size,
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "args": {
                    "warmup_steps": "auto",
                    "decay_until_step": "auto",
                    "max_lr": 1e-3,
                    "min_lr": 1e-5,
                },
            },
            "trainer": {
                "epochs": 10,
                "accumulation_steps": 2,
                "lr_schedule_per_step": True,
            },
        },
    )

    assert observed["scheduler"].decay_until_step == 10 * expected_steps_per_epoch
    assert observed["trained"] is True


def test_auto_per_step_scheduler_sums_variable_virtual_epoch_plan_under_ddp(
    monkeypatch,
):
    dataset = _MultiSrDataset([16000] * 8 + [8000] * 8 + [48000] * 8)
    sampler = GroupedBatchSampler(
        dataset,
        batch_size=2,
        rank=0,
        world_size=2,
        seed=34,
        num_per_epoch=10,
        drop_last=True,
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    observed = {}

    class TrainerSpy:
        def __init__(self, **kwargs):
            self.train_sampler = kwargs["train_sampler"]
            self.train_loader = kwargs["train_loader"]
            observed["scheduler"] = kwargs["scheduler"]
            observed["prepared_epoch"] = self.train_sampler.epoch
            observed["prepared_len"] = len(self.train_loader)

        def train(self):
            epoch_batches = []
            optimizer_steps = 0
            for epoch in range(1, 7):
                self.train_sampler.set_epoch(epoch)
                batches = list(self.train_loader)
                epoch_batches.append(len(batches))
                optimizer_steps += len(batches) // 2
            observed["epoch_batches"] = epoch_batches
            observed["optimizer_steps"] = optimizer_steps

    entry = _entry_spy()
    monkeypatch.setattr(
        "core.losses.build.build_loss", lambda *_args, **_kwargs: "loss"
    )
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)
    monkeypatch.setattr("torch.distributed.all_reduce", lambda *_args, **_kwargs: None)

    entry.run_training(
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "args": {
                    "warmup_steps": "auto",
                    "decay_until_step": "auto",
                    "max_lr": 1e-3,
                    "min_lr": 1e-5,
                },
            },
            "trainer": {
                "epochs": 6,
                "accumulation_steps": 2,
                "lr_schedule_per_step": True,
            },
        },
        args=Namespace(
            resume=False, resume_from=None, wandb=False, checkpoint=None
        ),
        rank=0,
        local_rank=0,
        world_size=2,
        train_loader=train_loader,
        val_loader=DataLoader([]),
        train_sampler=sampler,
        train_ds=dataset,
        model=model,
        trainer_cls=TrainerSpy,
    )

    assert observed == {
        "scheduler": observed["scheduler"],
        "prepared_epoch": 1,
        "prepared_len": 2,
        "epoch_batches": [2, 3, 1, 3, 2, 2],
        "optimizer_steps": 5,
    }
    assert observed["scheduler"].decay_until_step == observed["optimizer_steps"]


def test_auto_per_step_scheduler_rejects_zero_optimizer_step_horizon(monkeypatch):
    entry = _entry_spy()

    with pytest.raises(ValueError, match="scheduler horizon is zero"):
        _run_minimal_scheduler_training(
            monkeypatch,
            entry,
            world_size=2,
            train_loader=DataLoader([0], batch_size=1),
            cfg={
                "data": {"sample_rate": 16000},
                "loss": {},
                "scheduler": {
                    "type": "LinearWarmupCosineAnnealing",
                    "args": {
                        "warmup_steps": "auto",
                        "decay_until_step": "auto",
                        "max_lr": 1e-3,
                        "min_lr": 1e-5,
                    },
                },
                "trainer": {
                    "epochs": 3,
                    "accumulation_steps": 2,
                    "lr_schedule_per_step": True,
                },
            },
        )


@pytest.mark.parametrize(
    ("start_epoch", "resume", "expected_batches"),
    [(1, False, 1), (7, True, 2)],
)
def test_real_auto_scheduler_uses_exact_cardinality_and_does_not_collapse_lr(
    monkeypatch, tmp_path, start_epoch, resume, expected_batches
):
    dataset = _MultiSrDataset()
    sampler = GroupedBatchSampler(
        dataset,
        batch_size=3,
        seed=1,
        num_per_epoch=7,
        drop_last=True,
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    observed = {}

    class TrainerSpy:
        def __init__(self, **kwargs):
            self.train_sampler = kwargs["train_sampler"]
            self.scheduler = kwargs["scheduler"]
            self.optimizer = kwargs["optimizer"]
            self.start_epoch = start_epoch

        def train(self):
            self.train_sampler.set_epoch(self.start_epoch)
            observed["epoch_batches"] = len(self.train_sampler)
            observed["decay_until_step"] = self.scheduler.decay_until_step
            observed["initial_lr"] = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            self.scheduler.step()
            observed["first_step_lr"] = self.optimizer.param_groups[0]["lr"]

    entry = _entry_spy()
    monkeypatch.setattr(
        "core.losses.build.build_loss", lambda *_args, **_kwargs: "loss"
    )
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)

    entry.run_training(
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "args": {
                    "warmup_steps": "auto",
                    "decay_until_step": "auto",
                    "max_lr": 1e-3,
                    "min_lr": 1e-5,
                },
                "auto": {"warmup_ratio": 0.1},
            },
            "trainer": {
                "epochs": 10,
                "accumulation_steps": 1,
                "lr_schedule_per_step": True,
                "exp_path": str(tmp_path / "run"),
            },
        },
        args=_resume_args(tmp_path, start_epoch=start_epoch, resume=resume),
        rank=0,
        local_rank=0,
        world_size=1,
        train_loader=train_loader,
        val_loader=DataLoader([]),
        train_sampler=sampler,
        train_ds=dataset,
        model=model,
        trainer_cls=TrainerSpy,
    )

    assert observed["epoch_batches"] == expected_batches
    expected_horizon = sum(
        optimizer_steps_per_epoch(
            sampler.batch_count_for_epoch(epoch), 1, world_size=1
        )
        for epoch in range(1, 11)
    )
    assert observed["decay_until_step"] == expected_horizon
    assert observed["initial_lr"] > 1e-5
    assert observed["first_step_lr"] > 1e-5


@pytest.mark.parametrize(
    ("start_epoch", "resume"),
    [(1, False), (7, True)],
)
@pytest.mark.parametrize(
    ("drop_last", "epoch_lengths"),
    [(False, {1: 3, 7: 3}), (True, {1: 1, 7: 2})],
)
def test_ddp_scheduler_uses_all_reduce_target_cardinality_without_collective(
    monkeypatch, tmp_path, start_epoch, resume, drop_last, epoch_lengths
):
    dataset = _MultiSrDataset()
    sampler = GroupedBatchSampler(
        dataset,
        batch_size=2,
        rank=1,
        world_size=2,
        seed=1,
        num_per_epoch=7,
        drop_last=drop_last,
    )
    train_loader = DataLoader(dataset, batch_sampler=sampler)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    observed = {}

    class TrainerSpy:
        def __init__(self, **kwargs):
            observed["scheduler"] = kwargs["scheduler"]
            observed["prepared_len"] = len(kwargs["train_loader"])

        def train(self):
            return None

    entry = _entry_spy()
    monkeypatch.setattr(
        "core.losses.build.build_loss", lambda *_args, **_kwargs: "loss"
    )
    monkeypatch.setattr(entry, "build_optimizer", lambda *_args: optimizer)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(entry, "build_stft_module", lambda *_args: None)
    monkeypatch.setattr(entry, "setup_signal_handlers", lambda *_args: None)
    collective_calls = []
    monkeypatch.setattr(
        "torch.distributed.all_reduce",
        lambda *_args, **_kwargs: collective_calls.append((_args, _kwargs)),
    )

    entry.run_training(
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "args": {
                    "warmup_steps": "auto",
                    "decay_until_step": "auto",
                    "max_lr": 1e-3,
                    "min_lr": 1e-5,
                },
            },
            "trainer": {
                "epochs": 10,
                "accumulation_steps": 1,
                "lr_schedule_per_step": True,
                "exp_path": str(tmp_path / "run"),
            },
        },
        args=_resume_args(tmp_path, start_epoch=start_epoch, resume=resume),
        rank=1,
        local_rank=1,
        world_size=2,
        train_loader=train_loader,
        val_loader=DataLoader([]),
        train_sampler=sampler,
        train_ds=dataset,
        model=model,
        trainer_cls=TrainerSpy,
    )

    expected_len = epoch_lengths[start_epoch]
    assert observed["prepared_len"] == expected_len
    expected_horizon = sum(
        optimizer_steps_per_epoch(
            sampler.batch_count_for_epoch(epoch), 1, world_size=2
        )
        for epoch in range(1, 11)
    )
    assert observed["scheduler"].decay_until_step == expected_horizon
    assert collective_calls == []


@pytest.mark.parametrize(("n_batches", "accumulation"), [(3, 2), (5, 3)])
@pytest.mark.parametrize(("world_size", "expected_steps"), [(1, 2), (2, 1)])
def test_scheduler_estimate_matches_actual_optimizer_tail_policy(
    n_batches, accumulation, world_size, expected_steps
):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    real_step = optimizer.step
    step_calls = 0

    def step_spy(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return real_step(*args, **kwargs)

    optimizer.step = step_spy
    full_windows, tail = divmod(n_batches, accumulation)
    for _ in range(full_windows):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if tail:
        model.weight.grad = torch.ones_like(model.weight)
        trainer = SimpleNamespace(
            world_size=world_size,
            rank=0,
            model=model,
            optimizer=optimizer,
            n_steps=step_calls,
            use_amp=False,
            amp_dtype=torch.float32,
            _approx_batch_size=1,
            _optimizer_step=optimizer.step,
            _log_train_loss_after_step=lambda: None,
        )
        MetricsCapableTrainer._handle_epoch_tail_gradients(
            trainer,
            tail,
            accumulation,
            epoch=1,
            zero_grad_fn=lambda: optimizer.zero_grad(set_to_none=True),
            step_fn=optimizer.step,
        )

    assert step_calls == expected_steps
    assert optimizer_steps_per_epoch(
        n_batches, accumulation, world_size=world_size
    ) == step_calls


def test_epoch_tail_handler_rejects_empty_accum_window():
    """accum_count=0 必须在入口报错，而不是在 scale_correction 处除零。"""
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        world_size=1,
        rank=0,
        model=model,
        optimizer=optimizer,
        n_steps=0,
        _approx_batch_size=1,
        _optimizer_step=optimizer.step,
        _log_train_loss_after_step=lambda: None,
    )
    with pytest.raises(ValueError, match="accum_count > 0"):
        MetricsCapableTrainer._handle_epoch_tail_gradients(trainer, 0, 2, epoch=1)


_GLOO_PREVIEW_TIMEOUT = 45


def _gloo_resume_preview_worker(
    rank,
    world_size,
    init_file,
    rank0_checkpoint,
    rank1_checkpoint,
    rank0_cwd,
    rank1_cwd,
    result_dir,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    resolved = None
    entry = _entry_spy()
    try:
        cwd = rank0_cwd if rank == 0 else rank1_cwd
        if cwd is not None:
            os.chdir(cwd)
        checkpoint = rank0_checkpoint if rank == 0 else rank1_checkpoint
        args = Namespace(resume=True, resume_from=checkpoint)
        sampler = GroupedBatchSampler(
            _MultiSrDataset(),
            batch_size=3,
            rank=rank,
            world_size=world_size,
            seed=1,
            num_per_epoch=7,
            drop_last=False,
        )
        try:
            prepared = entry._prepare_train_sampler_for_scheduler(
                {"trainer": {"exp_path": str(Path(result_dir) / "run")}},
                args,
                sampler,
            )
            resolved = prepared.snapshot_path
            entry._validate_resume_checkpoint_before_trainer(resolved)
            outcome = ("ok", str(resolved), prepared.start_epoch, sampler.epoch)
        except Exception as exc:
            outcome = ("error", type(exc).__name__, str(exc))

        gathered = [None] * world_size
        dist.all_gather_object(gathered, outcome)
        entry._cleanup_resume_checkpoint_snapshot(resolved)
        Path(result_dir, f"rank{rank}.txt").write_text(
            repr(gathered), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def _gloo_rank_local_resume_snapshot_worker(
    rank, world_size, init_file, rank0_checkpoint, rank1_checkpoint, result_dir
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    entry = _entry_spy()
    resolved = None
    try:
        rank_dir = Path(result_dir) / f"rank{rank}"
        checkpoint = rank0_checkpoint if rank == 0 else rank1_checkpoint
        prepared = entry._prepare_train_sampler_for_scheduler(
            {"trainer": {"exp_path": str(rank_dir / "run")}},
            Namespace(resume=True, resume_from=checkpoint),
            None,
        )
        resolved = prepared.snapshot_path
        entry._validate_resume_checkpoint_before_trainer(resolved)
        outcome = (
            "ok",
            str(resolved),
            prepared.start_epoch,
            entry._checkpoint_content_identity(resolved),
        )
        gathered = [None] * world_size
        dist.all_gather_object(gathered, outcome)
        if rank == 0:
            Path(result_dir, "rank-local-snapshots.txt").write_text(
                repr(gathered), encoding="utf-8"
            )
    finally:
        entry._cleanup_resume_checkpoint_snapshot(resolved)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_resume_uses_rank_local_snapshots_with_matching_content(tmp_path):
    rank0_checkpoint = tmp_path / "rank0" / "latest_model.tar"
    rank1_checkpoint = tmp_path / "rank1" / "renamed-checkpoint.pth"
    rank0_checkpoint.parent.mkdir()
    rank1_checkpoint.parent.mkdir()
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, rank0_checkpoint)
    rank1_checkpoint.write_bytes(rank0_checkpoint.read_bytes())
    assert rank0_checkpoint.stat().st_ino != rank1_checkpoint.stat().st_ino

    mp.spawn(
        _gloo_rank_local_resume_snapshot_worker,
        args=(
            2,
            str(tmp_path / "rank_local_init"),
            str(rank0_checkpoint),
            str(rank1_checkpoint),
            str(tmp_path),
        ),
        nprocs=2,
        join=True,
    )

    first, second = ast.literal_eval(
        (tmp_path / "rank-local-snapshots.txt").read_text(encoding="utf-8")
    )
    assert first[0] == second[0] == "ok"
    assert first[1] != second[1]
    assert Path(first[1]).parent == (
        tmp_path / "rank0" / "run" / "resume-staging" / "rank-0"
    )
    assert Path(second[1]).parent == (
        tmp_path / "rank1" / "run" / "resume-staging" / "rank-1"
    )
    assert first[2] == second[2] == 7
    assert first[3] == second[3]
    assert not Path(first[1]).exists()
    assert not Path(second[1]).exists()


def _spawn_gloo_resume_preview(
    tmp_path,
    rank0_checkpoint,
    rank1_checkpoint,
    *,
    rank0_cwd=None,
    rank1_cwd=None,
):
    context = mp.spawn(
        _gloo_resume_preview_worker,
        args=(
            2,
            str(tmp_path / "preview_init"),
            str(rank0_checkpoint),
            str(rank1_checkpoint),
            str(rank0_cwd) if rank0_cwd is not None else None,
            str(rank1_cwd) if rank1_cwd is not None else None,
            str(tmp_path),
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
    while not context.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("Gloo resume preview test timed out")
    return (tmp_path / "rank0.txt").read_text(encoding="utf-8")


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_resume_preview_accepts_rank_local_paths_with_same_content(tmp_path):
    rank0_checkpoint = tmp_path / "rank0-middle.pth"
    rank1_checkpoint = tmp_path / "rank1-middle.pth"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, rank0_checkpoint)
    rank1_checkpoint.write_bytes(rank0_checkpoint.read_bytes())

    gathered = _spawn_gloo_resume_preview(
        tmp_path, rank0_checkpoint, rank1_checkpoint
    )

    first, second = ast.literal_eval(gathered)
    assert first[0] == second[0] == "ok"
    assert first[1] != second[1]
    assert Path(first[1]).is_absolute()
    assert Path(second[1]).is_absolute()
    assert first[2:] == second[2:] == (7, 7)
    assert not Path(first[1]).exists()
    assert not Path(second[1]).exists()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_resume_preview_aggregates_rank_local_not_found(tmp_path):
    rank0_checkpoint = tmp_path / "rank0-missing" / "model_E0007_B000001.pth"
    rank1_checkpoint = tmp_path / "rank1-valid.pth"
    torch.save({"epoch": 11, "n_steps": 100, "batch_cursor": 1}, rank1_checkpoint)

    gathered = _spawn_gloo_resume_preview(
        tmp_path, rank0_checkpoint, rank1_checkpoint
    )

    first, second = ast.literal_eval(gathered)
    assert first == second
    assert first[:2] == ("error", "RuntimeError")
    assert "rank 0: FileNotFoundError" in first[2]


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_resume_preview_resolves_each_rank_local_cwd(tmp_path):
    rank0_cwd = tmp_path / "rank0-cwd"
    rank1_cwd = tmp_path / "rank1-cwd"
    rank0_cwd.mkdir()
    rank1_cwd.mkdir()
    rank0_checkpoint = rank0_cwd / "checkpoints" / "latest_model.tar"
    rank1_checkpoint = rank1_cwd / "checkpoints" / "latest_model.tar"
    rank0_checkpoint.parent.mkdir()
    rank1_checkpoint.parent.mkdir()
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, rank0_checkpoint)
    rank1_checkpoint.write_bytes(rank0_checkpoint.read_bytes())

    gathered = _spawn_gloo_resume_preview(
        tmp_path,
        "checkpoints/latest_model.tar",
        "checkpoints/latest_model.tar",
        rank0_cwd=rank0_cwd,
        rank1_cwd=rank1_cwd,
    )

    first, second = ast.literal_eval(gathered)
    assert first[0] == second[0] == "ok"
    assert first[1] != second[1]
    assert first[2:] == second[2:] == (7, 7)
    assert not Path(first[1]).exists()
    assert not Path(second[1]).exists()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_resume_preview_aggregates_rank_local_parse_error(tmp_path):
    rank0_checkpoint = tmp_path / "rank0-corrupt.pth"
    rank0_checkpoint.write_bytes(b"not a torch checkpoint")
    rank1_checkpoint = tmp_path / "rank1-valid.pth"
    torch.save({"epoch": 11, "n_steps": 100, "batch_cursor": 1}, rank1_checkpoint)

    gathered = _spawn_gloo_resume_preview(
        tmp_path, rank0_checkpoint, rank1_checkpoint
    )

    assert gathered.startswith("[('error', 'RuntimeError', ")
    first, second = ast.literal_eval(gathered)
    assert first == second
    assert "rank 0: UnpicklingError" in first[2]


def _gloo_resume_load_lifecycle_worker(
    rank, world_size, init_file, checkpoint, replacement, result_dir
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    entry = _entry_spy()
    original_prepare = entry._prepare_train_sampler_for_scheduler

    def prepare_then_replace(cfg, args, train_sampler):
        result = original_prepare(cfg, args, train_sampler)
        if rank == 0:
            os.replace(replacement, checkpoint)
        dist.barrier()
        return result

    rank0_load_done = Path(result_dir) / "load-rank0.done"

    class TrainerSpy:
        def __init__(self, **kwargs):
            if rank == 1:
                # rank1 晚于 rank0 加载冻结 snapshot，用 rank0 的完成标记做确定性
                # 排序；超时显式失败，避免固定 sleep 在慢机器上变成静默通过。
                deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
                while not rank0_load_done.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "rank 0 did not finish loading the resume snapshot"
                        )
                    time.sleep(0.01)
            snapshot = Path(kwargs["resume_from"])
            loaded_epoch = safe_torch_load(snapshot)["epoch"]
            Path(result_dir, f"load-rank{rank}.txt").write_text(
                repr((loaded_epoch, str(snapshot), snapshot.exists())), encoding="utf-8"
            )
            if rank == 0:
                rank0_load_done.write_text(str(snapshot), encoding="utf-8")

        def train(self):
            return None

    entry._prepare_train_sampler_for_scheduler = prepare_then_replace
    model, optimizer = _tiny_model_optimizer()
    entry.build_optimizer = lambda *_args: optimizer
    entry.build_scheduler = lambda *_args, **_kwargs: "scheduler"
    entry.build_stft_module = lambda *_args: None
    entry.setup_signal_handlers = lambda *_args: None
    import core.losses.build as loss_build

    loss_build.build_loss = lambda *_args, **_kwargs: "loss"
    entry.run_training(
        cfg={
            "data": {"sample_rate": 16000},
            "loss": {},
            "trainer": {"exp_path": str(Path(result_dir) / "run")},
        },
        args=Namespace(
            resume=True,
            resume_from=checkpoint,
            wandb=False,
            checkpoint=None,
        ),
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        train_loader=DataLoader([0]),
        val_loader=DataLoader([]),
        train_sampler=None,
        train_ds=None,
        model=model,
        trainer_cls=TrainerSpy,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_resume_snapshot_survives_until_all_trainers_finish_loading(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    replacement = tmp_path / "replacement.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    torch.save({"epoch": 12, "n_steps": 100, "batch_cursor": 1}, replacement)

    mp.spawn(
        _gloo_resume_load_lifecycle_worker,
        args=(
            2,
            str(tmp_path / "lifecycle_init"),
            str(checkpoint),
            str(replacement),
            str(tmp_path),
        ),
        nprocs=2,
        join=True,
    )

    first = ast.literal_eval(
        (tmp_path / "load-rank0.txt").read_text(encoding="utf-8")
    )
    second = ast.literal_eval(
        (tmp_path / "load-rank1.txt").read_text(encoding="utf-8")
    )
    assert first[0] == second[0] == 7
    assert first[1] != second[1]
    assert first[2] is second[2] is True
    assert not Path(first[1]).exists()
    assert not Path(second[1]).exists()
    assert safe_torch_load(checkpoint)["epoch"] == 12


def _gloo_sampler_stage_failure_worker(
    rank, world_size, init_file, checkpoint, result_dir
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    entry = _entry_spy()

    failed_rank = 1 if "rank1" in Path(checkpoint).name else 0

    class SamplerSpy:
        def set_epoch(self, _epoch):
            if rank == failed_rank:
                raise RuntimeError(f"injected sampler rank {failed_rank} failure")

    try:
        entry._prepare_train_sampler_for_scheduler(
            {"trainer": {"exp_path": str(Path(result_dir) / "run")}},
            Namespace(resume=True, resume_from=checkpoint),
            SamplerSpy(),
        )
        outcome = ("ok",)
    except Exception as exc:
        outcome = ("error", type(exc).__name__, str(exc))
    finally:
        Path(result_dir, f"sampler-rank{rank}.txt").write_text(
            repr(outcome), encoding="utf-8"
        )
        dist.destroy_process_group()


def _gloo_original_trainer_lifecycle_worker(
    rank, world_size, init_file, checkpoint, result_dir
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    entry = _entry_spy()
    import core.trainer.base_trainer as base_trainer_module

    class TestDDP(torch.nn.Module):
        def __init__(self, module, **_kwargs):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    base_trainer_module.DDP = TestDDP
    loaded_paths = []

    class OriginalLifecycleTrainer(BaseTrainer):
        def __init__(
            self,
            train_loader,
            val_loader,
            sample_rate=16000,
            stft_module=None,
            **kwargs,
        ):
            self.resume_hook_before_super = (
                type(self)._resume_checkpoint is BaseTrainer._resume_checkpoint
            )
            self.resume_hook_world_size = None
            self.loaded_checkpoint_paths = []
            super().__init__(**kwargs)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.nested_alias = {"active_model": self.model}
            self.post_super_observation = (
                self.start_epoch,
                self.n_steps,
                self.resume_hook_world_size,
            )

        @staticmethod
        def _safe_load(path, **kwargs):
            loaded_paths.append(str(Path(path)))
            return BaseTrainer._safe_load(path, **kwargs)

        def _after_resume_checkpoint(self, checkpoint_state):
            self.resume_hook_world_size = self.world_size

        def train(self):
            if rank == 0:
                self._save_checkpoint(8, skip_barrier=True)
            Path(result_dir, f"original-lifecycle-rank{rank}.txt").write_text(
                repr(
                    (
                        type(self) is OriginalLifecycleTrainer,
                        self.resume_hook_before_super,
                        self._resume_checkpoint.__func__
                        is BaseTrainer._resume_checkpoint,
                        self.post_super_observation,
                        self.nested_alias["active_model"] is self.model,
                        str(self.ckpt_dir),
                        str(self.log_dir),
                        tuple(loaded_paths),
                        str(self._resume_from),
                        str(self._resume_load_from),
                    )
                ),
                encoding="utf-8",
            )

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    entry.build_optimizer = lambda *_args: optimizer
    entry.build_scheduler = lambda *_args, **_kwargs: scheduler
    entry.build_stft_module = lambda *_args: None
    entry.setup_signal_handlers = lambda *_args: None
    import core.losses.build as loss_build

    loss_build.build_loss = lambda *_args, **_kwargs: "loss"
    try:
        entry.run_training(
            cfg={
                "data": {"sample_rate": 16000},
                "loss": {},
                "trainer": {
                    "exp_path": str(Path(result_dir) / "original-lifecycle-exp"),
                    "validation_epoch_interval": 0,
                    "save_top_k": 0,
                },
            },
            args=Namespace(
                resume=True,
                resume_from=checkpoint,
                wandb=False,
                checkpoint=None,
            ),
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            train_loader=DataLoader([0]),
            val_loader=DataLoader([]),
            train_sampler=None,
            train_ds=None,
            model=model,
            trainer_cls=OriginalLifecycleTrainer,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _gloo_post_super_failure_worker(
    rank, world_size, init_file, checkpoint, result_dir, failed_rank
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    entry = _entry_spy()
    import core.trainer.base_trainer as base_trainer_module

    class TestDDP(torch.nn.Module):
        def __init__(self, module, **_kwargs):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    base_trainer_module.DDP = TestDDP

    class PostSuperFailureTrainer(BaseTrainer):
        def __init__(
            self,
            train_loader,
            val_loader,
            sample_rate=16000,
            stft_module=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.train_loader = train_loader
            self.val_loader = val_loader

            def reject_post_super_collective(*_args, **_kwargs):
                raise AssertionError("entry invoked a post-super distributed collective")

            dist.barrier = reject_post_super_collective
            dist.broadcast_object_list = reject_post_super_collective
            dist.all_gather_object = reject_post_super_collective
            if rank == failed_rank:
                raise RuntimeError(f"injected post-super rank {rank} failure")

        def train(self):
            return None

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    entry.build_optimizer = lambda *_args: optimizer
    entry.build_scheduler = lambda *_args, **_kwargs: scheduler
    entry.build_stft_module = lambda *_args: None
    entry.setup_signal_handlers = lambda *_args: None
    import core.losses.build as loss_build

    loss_build.build_loss = lambda *_args, **_kwargs: "loss"
    try:
        entry.run_training(
            cfg={
                "data": {"sample_rate": 16000},
                "loss": {},
                "trainer": {
                    "exp_path": str(Path(result_dir) / "post-super-exp"),
                    "validation_epoch_interval": 0,
                    "save_top_k": 0,
                },
            },
            args=Namespace(
                resume=True,
                resume_from=checkpoint,
                wandb=False,
                checkpoint=None,
            ),
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            train_loader=DataLoader([0]),
            val_loader=DataLoader([]),
            train_sampler=None,
            train_ds=None,
            model=model,
            trainer_cls=PostSuperFailureTrainer,
        )
        outcome = ("ok",)
    except Exception as exc:
        outcome = ("error", type(exc).__name__, str(exc))
    finally:
        Path(result_dir, f"post-super-{failed_rank}-rank{rank}.txt").write_text(
            repr(outcome), encoding="utf-8"
        )
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
@pytest.mark.parametrize("failed_rank", [0, 1])
def test_gloo_post_super_failure_preserves_peer_and_cleans_shared_snapshot(
    tmp_path, failed_rank
):
    checkpoint = tmp_path / f"post-super-{failed_rank}-latest_model.tar"
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    torch.save(
        {
            "epoch": 7,
            "batch_cursor": 1,
            "n_steps": 3,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint,
    )

    context = mp.spawn(
        _gloo_post_super_failure_worker,
        args=(
            2,
            str(tmp_path / f"post_super_{failed_rank}_init"),
            str(checkpoint),
            str(tmp_path),
            failed_rank,
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
    while not context.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("Gloo post-super failure test timed out")

    outcomes = [
        ast.literal_eval(
            (tmp_path / f"post-super-{failed_rank}-rank{rank}.txt").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    ]
    assert outcomes[failed_rank] == (
        "error",
        "RuntimeError",
        f"injected post-super rank {failed_rank} failure",
    )
    assert outcomes[1 - failed_rank] == ("ok",)
    assert not list(tmp_path.glob(f".{checkpoint.name}.resume-*"))


def test_last_resume_consumer_cleans_snapshot_and_sidecar(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    entry = _entry_spy()
    snapshot, identity = entry._freeze_resume_checkpoint(checkpoint)
    entry._resolved_resume_identity = identity

    entry._acquire_resume_checkpoint_consumer(snapshot)
    entry._complete_resume_checkpoint_consumer_acquisition(snapshot)
    assert snapshot.exists()
    assert entry._resume_snapshot_lock_path(snapshot).exists()

    entry._release_resume_checkpoint_consumer(snapshot)

    assert not snapshot.exists()
    assert not entry._resume_snapshot_lock_path(snapshot).exists()


def test_resume_preview_collective_exception_releases_local_snapshot(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    entry = _entry_spy()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected preview collective failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected preview collective failure"):
        entry._prepare_train_sampler_for_scheduler(
            {"trainer": {"exp_path": str(tmp_path / "run")}},
            Namespace(resume=True, resume_from=str(checkpoint)),
            None,
        )

    assert not getattr(entry, "_resume_snapshot_leases", {})
    assert not list((tmp_path / "run" / "resume-staging").glob("**/*.snapshot"))


def test_resume_validation_collective_exception_cleans_local_owner(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "validation-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    entry = _entry_spy()
    snapshot, identity = entry._freeze_resume_checkpoint(checkpoint)
    sidecar = entry._resume_snapshot_lock_path(snapshot)
    entry._resolved_resume_identity = identity
    original_cleanup = entry._cleanup_resume_checkpoint_snapshot

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected validation collective failure")
        ),
    )

    def cleanup_then_fail(*args, **kwargs):
        original_cleanup(*args, **kwargs)
        raise RuntimeError("injected validation cleanup failure")

    monkeypatch.setattr(entry, "_cleanup_resume_checkpoint_snapshot", cleanup_then_fail)

    with pytest.raises(
        RuntimeError, match="injected validation collective failure"
    ):
        entry._validate_resume_checkpoint_before_trainer(snapshot)

    assert not snapshot.exists()
    assert not sidecar.exists()
    assert not getattr(entry, "_resume_snapshot_leases", {})


def test_distributed_stage_collective_exception_cleans_local_owner(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "stage-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    entry = _entry_spy()
    snapshot, _identity = entry._freeze_resume_checkpoint(checkpoint)
    sidecar = entry._resume_snapshot_lock_path(snapshot)
    original_cleanup = entry._cleanup_resume_checkpoint_snapshot

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected stage collective failure")
        ),
    )

    def cleanup_then_fail(*args, **kwargs):
        original_cleanup(*args, **kwargs)
        raise RuntimeError("injected stage cleanup failure")

    monkeypatch.setattr(entry, "_cleanup_resume_checkpoint_snapshot", cleanup_then_fail)

    with pytest.raises(RuntimeError, match="injected stage collective failure"):
        entry._run_distributed_stage(
            "sampler preparation",
            lambda: None,
            checkpoint_path=snapshot,
        )

    assert not snapshot.exists()
    assert not sidecar.exists()
    assert not getattr(entry, "_resume_snapshot_leases", {})


@pytest.mark.parametrize("failure_gather", [1, 2])
def test_consumer_collective_exception_releases_local_lease(
    monkeypatch, tmp_path, failure_gather
):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    entry = _entry_spy()
    snapshot, identity = entry._freeze_resume_checkpoint(checkpoint)
    lease = entry._resume_snapshot_leases[snapshot]
    entry._resolved_resume_identity = identity
    entry._resolved_resume_lock_identity = lease[4]
    gather_calls = 0

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather_or_fail(output, value):
        nonlocal gather_calls
        gather_calls += 1
        if gather_calls == failure_gather:
            raise RuntimeError(f"injected consumer gather {failure_gather} failure")
        output[:] = [value, value]

    monkeypatch.setattr(dist, "all_gather_object", gather_or_fail)

    with pytest.raises(
        RuntimeError, match=f"injected consumer gather {failure_gather} failure"
    ):
        entry._prepare_resume_checkpoint_consumers(snapshot)

    assert not getattr(entry, "_resume_snapshot_leases", {})
    assert not getattr(entry, "_resume_snapshot_consumers", {})
    assert not getattr(entry, "_resume_snapshot_transition_guards", {})
    assert not snapshot.exists()
    assert not entry._resume_snapshot_lock_path(snapshot).exists()


def _gloo_consumer_acquisition_failure_worker(
    rank,
    world_size,
    init_file,
    checkpoint,
    result_dir,
    failed_rank,
    inject_unlock_failure=False,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    entry = _entry_spy()
    trainer_entered = False
    acquisition_started = False
    import core.trainer.entry as entry_module

    original_register = entry_module.atexit.register
    original_all_gather = dist.all_gather_object
    original_flock = entry_module.fcntl.flock
    unlock_failed = False

    def inject_consumer_registration_failure(callback, *args, **kwargs):
        nonlocal acquisition_started
        callback_name = getattr(getattr(callback, "func", None), "__name__", "")
        if callback_name == "_atexit_release_resume_checkpoint_consumer":
            acquisition_started = True
            if failed_rank == "all" or rank == failed_rank:
                raise RuntimeError(
                    f"injected consumer rank {rank} registration failure"
                )
        return original_register(callback, *args, **kwargs)

    def reject_collective(*_args, **_kwargs):
        raise AssertionError("entry invoked collective after acquisition gather")

    def fail_first_transition_unlock(fd, operation):
        nonlocal unlock_failed
        if (
            inject_unlock_failure
            and rank == 0
            and operation == fcntl.LOCK_UN
            and not unlock_failed
        ):
            unlock_failed = True
            raise InterruptedError("injected transition guard unlock failure")
        return original_flock(fd, operation)

    def gather_then_reject(*args, **kwargs):
        result = original_all_gather(*args, **kwargs)
        if acquisition_started:
            dist.barrier = reject_collective
            dist.broadcast_object_list = reject_collective
            dist.all_gather_object = reject_collective
            entry_module.fcntl.flock = fail_first_transition_unlock
        return result

    entry_module.atexit.register = inject_consumer_registration_failure
    dist.all_gather_object = gather_then_reject
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    entry.build_optimizer = lambda *_args: optimizer
    entry.build_scheduler = lambda *_args, **_kwargs: scheduler
    entry.build_stft_module = lambda *_args: None
    entry.setup_signal_handlers = lambda *_args: None
    import core.losses.build as loss_build

    loss_build.build_loss = lambda *_args, **_kwargs: "loss"

    class TrainerMustNotRun:
        def __init__(self, **_kwargs):
            nonlocal trainer_entered
            trainer_entered = True
            raise AssertionError("trainer entered after consumer acquisition failure")

    try:
        entry.run_training(
            cfg={
                "data": {"sample_rate": 16000},
                "loss": {},
                "trainer": {"exp_path": str(Path(result_dir) / "run")},
            },
            args=Namespace(
                resume=True,
                resume_from=checkpoint,
                wandb=False,
                checkpoint=None,
            ),
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            train_loader=DataLoader([0]),
            val_loader=DataLoader([]),
            train_sampler=None,
            train_ds=None,
            model=model,
            trainer_cls=TrainerMustNotRun,
        )
        result = ("ok",)
    except Exception as exc:
        result = ("error", type(exc).__name__, str(exc))
    finally:
        entry_module.atexit.register = original_register
        entry_module.fcntl.flock = original_flock
    state = (
        bool(getattr(entry, "_resume_snapshot_leases", {})),
        bool(getattr(entry, "_resume_snapshot_consumers", {})),
        bool(getattr(entry, "_resume_snapshot_transition_guards", {})),
        trainer_entered,
    )
    Path(result_dir, f"consumer-acquire-{failed_rank}-rank{rank}.txt").write_text(
        repr((result, state)), encoding="utf-8"
    )
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
@pytest.mark.parametrize("failed_rank", [0, 1, "all"])
def test_gloo_consumer_acquisition_failure_is_aggregated_and_cleaned(
    tmp_path, failed_rank
):
    checkpoint = tmp_path / f"consumer-acquire-{failed_rank}-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    mp.spawn(
        _gloo_consumer_acquisition_failure_worker,
        args=(
            2,
            str(tmp_path / f"consumer_acquire_{failed_rank}_init"),
            str(checkpoint),
            str(tmp_path),
            failed_rank,
        ),
        nprocs=2,
        join=True,
    )

    outcomes = [
        ast.literal_eval(
            (tmp_path / f"consumer-acquire-{failed_rank}-rank{rank}.txt").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    ]
    results = [outcome[0] for outcome in outcomes]
    states = [outcome[1] for outcome in outcomes]
    assert results[0] == results[1]
    assert results[0][:2] == ("error", "RuntimeError")
    assert states == [
        (False, False, False, False),
        (False, False, False, False),
    ]
    failed_ranks = range(2) if failed_rank == "all" else [failed_rank]
    for rank in failed_ranks:
        assert (
            f"rank {rank}: RuntimeError: injected consumer rank {rank} "
            "registration failure"
        ) in results[0][2]
    assert not list(tmp_path.glob(f".{checkpoint.name}.resume-*"))
    guard_paths = sorted(
        tmp_path.glob(
            f"run/resume-staging/rank-*/.{checkpoint.name}.resume.guard.lock"
        )
    )
    assert len(guard_paths) == 2
    for guard_path in guard_paths:
        guard_fd = os.open(guard_path, os.O_RDWR)
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(guard_fd, fcntl.LOCK_UN)
        finally:
            os.close(guard_fd)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_acquisition_guard_unlock_failure_preserves_original_aggregated_error(
    tmp_path,
):
    checkpoint = tmp_path / "consumer-acquire-unlock-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    mp.spawn(
        _gloo_consumer_acquisition_failure_worker,
        args=(
            2,
            str(tmp_path / "consumer_acquire_unlock_init"),
            str(checkpoint),
            str(tmp_path),
            "all",
            True,
        ),
        nprocs=2,
        join=True,
    )

    outcomes = [
        ast.literal_eval(
            (tmp_path / f"consumer-acquire-all-rank{rank}.txt").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    ]
    results = [outcome[0] for outcome in outcomes]
    states = [outcome[1] for outcome in outcomes]
    assert results[0] == results[1]
    assert results[0][:2] == ("error", "RuntimeError")
    assert "injected transition guard unlock failure" not in results[0][2]
    assert states == [
        (False, False, False, False),
        (False, False, False, False),
    ]
    assert not list(tmp_path.glob(f".{checkpoint.name}.resume-*"))


def _gloo_consumer_completion_failure_worker(
    rank, world_size, init_file, checkpoint, result_dir, fault_kind
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_GLOO_PREVIEW_TIMEOUT - 10),
    )
    entry = _entry_spy()
    trainer_entered = False
    consumer_registered = False
    fault_injected = False
    guard_close_attempts = 0
    completion_gather_count = 0
    target_guard_fd = None
    import core.trainer.entry as entry_module

    original_register = entry_module.atexit.register
    original_all_gather = dist.all_gather_object
    original_flock = entry_module.fcntl.flock
    original_close = entry_module.os.close
    collective_names = (
        "barrier",
        "monitored_barrier",
        "broadcast",
        "broadcast_object_list",
        "_broadcast_coalesced",
        "all_reduce",
        "all_reduce_coalesced",
        "reduce",
        "all_gather",
        "all_gather_coalesced",
        "all_gather_into_tensor",
        "all_gather_object",
        "_all_gather_base",
        "gather",
        "gather_object",
        "scatter",
        "scatter_object_list",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "_reduce_scatter_base",
        "all_to_all",
        "all_to_all_single",
    )
    original_collectives = {
        name: getattr(dist, name) for name in collective_names if hasattr(dist, name)
    }
    collective_calls = []

    class CollectiveAfterCompletionError(BaseException):
        pass

    def track_consumer_registration(callback, *args, **kwargs):
        nonlocal consumer_registered
        callback_name = getattr(getattr(callback, "func", None), "__name__", "")
        if callback_name == "_atexit_release_resume_checkpoint_consumer":
            consumer_registered = True
        return original_register(callback, *args, **kwargs)

    def reject_collective(name):
        def reject(*_args, **_kwargs):
            collective_calls.append(name)
            raise CollectiveAfterCompletionError(
                f"entry invoked {name} after completion gather"
            )

        return reject

    def fail_transition_unlock(fd, operation):
        nonlocal fault_injected
        if (
            rank == 0
            and fault_kind == "unlock"
            and operation == fcntl.LOCK_UN
            and not fault_injected
        ):
            fault_injected = True
            raise InterruptedError("injected successful-acquisition guard unlock failure")
        return original_flock(fd, operation)

    def fail_transition_close(fd):
        nonlocal fault_injected, guard_close_attempts
        if rank == 0 and fault_kind == "close" and fd == target_guard_fd:
            guard_close_attempts += 1
            fault_injected = True
            raise InterruptedError("injected successful-acquisition guard close failure")
        return original_close(fd)

    def gather_then_install_fault(*args, **kwargs):
        nonlocal completion_gather_count, target_guard_fd
        result = original_all_gather(*args, **kwargs)
        if consumer_registered and completion_gather_count == 0:
            completion_gather_count = 1
            guards = getattr(entry, "_resume_snapshot_transition_guards", {})
            target_guard_fd = next(iter(guards.values()), None)
            entry_module.fcntl.flock = fail_transition_unlock
            entry_module.os.close = fail_transition_close
        elif completion_gather_count == 1:
            completion_gather_count = 2
            for name in original_collectives:
                setattr(dist, name, reject_collective(name))
        return result

    entry_module.atexit.register = track_consumer_registration
    dist.all_gather_object = gather_then_install_fault
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    entry.build_optimizer = lambda *_args: optimizer
    entry.build_scheduler = lambda *_args, **_kwargs: scheduler
    entry.build_stft_module = lambda *_args: None
    entry.setup_signal_handlers = lambda *_args: None
    import core.losses.build as loss_build

    loss_build.build_loss = lambda *_args, **_kwargs: "loss"

    class TrainerMustNotRun:
        def __init__(self, **_kwargs):
            nonlocal trainer_entered
            trainer_entered = True
            raise AssertionError("trainer entered after consumer completion failure")

    try:
        entry.run_training(
            cfg={
                "data": {"sample_rate": 16000},
                "loss": {},
                "trainer": {"exp_path": str(Path(result_dir) / "run")},
            },
            args=Namespace(
                resume=True,
                resume_from=checkpoint,
                wandb=False,
                checkpoint=None,
            ),
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            train_loader=DataLoader([0]),
            val_loader=DataLoader([]),
            train_sampler=None,
            train_ds=None,
            model=model,
            trainer_cls=TrainerMustNotRun,
        )
        result = ("ok",)
    except (Exception, CollectiveAfterCompletionError) as exc:
        result = ("error", type(exc).__name__, str(exc))
    finally:
        entry_module.atexit.register = original_register
        entry_module.fcntl.flock = original_flock
        entry_module.os.close = original_close
        for name, collective in original_collectives.items():
            setattr(dist, name, collective)
    state = (
        bool(getattr(entry, "_resume_snapshot_leases", {})),
        bool(getattr(entry, "_resume_snapshot_consumers", {})),
        bool(getattr(entry, "_resume_snapshot_transition_guards", {})),
        trainer_entered,
        fault_injected,
        guard_close_attempts,
        tuple(collective_calls),
    )
    result_path = Path(
        result_dir, f"consumer-complete-{fault_kind}-rank{rank}.txt"
    )
    temporary_result_path = result_path.with_suffix(".tmp")
    temporary_result_path.write_text(repr((result, state)), encoding="utf-8")
    os.replace(temporary_result_path, result_path)
    release_path = Path(result_dir, f"consumer-complete-{fault_kind}.release")
    deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
    while not release_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("parent did not release completion-failure worker")
        time.sleep(0.01)
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
@pytest.mark.parametrize("fault_kind", ["unlock", "close"])
def test_gloo_consumer_completion_failure_is_aggregated_and_cleaned(
    tmp_path, fault_kind
):
    checkpoint = tmp_path / f"consumer-complete-{fault_kind}-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    context = mp.spawn(
        _gloo_consumer_completion_failure_worker,
        args=(
            2,
            str(tmp_path / f"consumer_complete_{fault_kind}_init"),
            str(checkpoint),
            str(tmp_path),
            fault_kind,
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
    result_paths = [
        tmp_path / f"consumer-complete-{fault_kind}-rank{rank}.txt"
        for rank in range(2)
    ]
    release_path = tmp_path / f"consumer-complete-{fault_kind}.release"
    try:
        while not all(path.exists() for path in result_paths):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Gloo consumer completion {fault_kind} test timed out"
                )
            time.sleep(0.01)

        assert all(process.is_alive() for process in context.processes)
        outcomes = [
            ast.literal_eval(path.read_text(encoding="utf-8"))
            for path in result_paths
        ]
        results = [outcome[0] for outcome in outcomes]
        states = [outcome[1] for outcome in outcomes]
        assert results[0] == results[1]
        assert results[0][:2] == ("error", "RuntimeError")
        assert (
            f"rank 0: InterruptedError: injected successful-acquisition guard "
            f"{fault_kind} failure"
        ) in results[0][2]
        expected_close_attempts = 1 if fault_kind == "close" else 0
        assert states == [
            (False, False, False, False, True, expected_close_attempts, ()),
            (False, False, False, False, False, 0, ()),
        ]
        assert not list(tmp_path.glob(f".{checkpoint.name}.resume-*"))
        guard_paths = sorted(
            tmp_path.glob(
                f"run/resume-staging/rank-*/.{checkpoint.name}.resume.guard.lock"
            )
        )
        assert len(guard_paths) == 2
        for guard_path in guard_paths:
            guard_fd = os.open(guard_path, os.O_RDWR)
            try:
                fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(guard_fd, fcntl.LOCK_UN)
            finally:
                os.close(guard_fd)
    finally:
        active_exception = sys.exc_info()[0] is not None
        release_path.touch()
        exit_deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
        while not context.join(max(0.0, exit_deadline - time.monotonic())):
            if time.monotonic() >= exit_deadline:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                if not active_exception:
                    raise TimeoutError(
                        f"Gloo consumer completion {fault_kind} workers did not exit"
                    )
                break


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_all_rank_consumer_acquisition_failure_preserves_concurrent_active_job(
    tmp_path,
):
    checkpoint = tmp_path / "consumer-acquire-active-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    holder = _entry_spy()
    active_snapshot, identity = holder._freeze_resume_checkpoint(checkpoint)
    holder._resolved_resume_identity = identity
    holder._acquire_resume_checkpoint_consumer(active_snapshot)
    holder._complete_resume_checkpoint_consumer_acquisition(active_snapshot)

    try:
        mp.spawn(
            _gloo_consumer_acquisition_failure_worker,
            args=(
                2,
                str(tmp_path / "consumer_acquire_active_init"),
                str(checkpoint),
                str(tmp_path),
                "all",
            ),
            nprocs=2,
            join=True,
        )
        outcomes = [
            ast.literal_eval(
                (tmp_path / f"consumer-acquire-all-rank{rank}.txt").read_text(
                    encoding="utf-8"
                )
            )
            for rank in range(2)
        ]
        results = [outcome[0] for outcome in outcomes]
        states = [outcome[1] for outcome in outcomes]
        assert results[0] == results[1]
        assert results[0][:2] == ("error", "RuntimeError")
        assert states == [
            (False, False, False, False),
            (False, False, False, False),
        ]
        assert active_snapshot.exists()
        assert holder._resume_snapshot_lock_path(active_snapshot).exists()
        assert sorted(tmp_path.glob(f".{checkpoint.name}.resume-*.snapshot")) == [
            active_snapshot
        ]
    finally:
        holder._release_resume_checkpoint_consumer(active_snapshot)

    assert not list(tmp_path.glob(f".{checkpoint.name}.resume-*"))


def test_consumer_atexit_cleans_last_snapshot(tmp_path):
    checkpoint = tmp_path / "atexit-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    ready = tmp_path / "atexit-ready.txt"
    script = """
from pathlib import Path
from tasks.aec.trainer.entry import AECTrainingEntry
entry = object.__new__(AECTrainingEntry)
snapshot, identity = entry._freeze_resume_checkpoint(Path({checkpoint!r}))
entry._resolved_resume_identity = identity
entry._acquire_resume_checkpoint_consumer(snapshot)
entry._complete_resume_checkpoint_consumer_acquisition(snapshot)
Path({ready!r}).write_text(str(snapshot), encoding='utf-8')
""".format(checkpoint=str(checkpoint), ready=str(ready))

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=_GLOO_PREVIEW_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    snapshot = Path(ready.read_text(encoding="utf-8"))
    assert not snapshot.exists()
    assert not _entry_spy()._resume_snapshot_lock_path(snapshot).exists()


def test_consumer_atexit_releases_owner_transition_guard(tmp_path):
    checkpoint = tmp_path / "atexit-transition-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    ready = tmp_path / "atexit-transition-ready.txt"
    script = """
from pathlib import Path
from tasks.aec.trainer.entry import AECTrainingEntry
entry = object.__new__(AECTrainingEntry)
snapshot, identity = entry._freeze_resume_checkpoint(Path({checkpoint!r}))
entry._resolved_resume_identity = identity
entry._acquire_resume_checkpoint_consumer(snapshot)
Path({ready!r}).write_text(str(snapshot), encoding='utf-8')
""".format(checkpoint=str(checkpoint), ready=str(ready))

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=_GLOO_PREVIEW_TIMEOUT,
    )

    assert result.returncode == 0, result.stderr
    snapshot = Path(ready.read_text(encoding="utf-8"))
    assert not snapshot.exists()
    assert not _entry_spy()._resume_snapshot_lock_path(snapshot).exists()


def test_stale_scan_preserves_active_shared_consumer(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    holder = _entry_spy()
    active_snapshot, identity = holder._freeze_resume_checkpoint(checkpoint)
    holder._resolved_resume_identity = identity
    holder._acquire_resume_checkpoint_consumer(active_snapshot)
    holder._complete_resume_checkpoint_consumer_acquisition(active_snapshot)

    scanner = _entry_spy()
    concurrent_snapshot, _identity = scanner._freeze_resume_checkpoint(checkpoint)
    try:
        assert active_snapshot.exists()
    finally:
        scanner._cleanup_resume_checkpoint_snapshot(concurrent_snapshot)
        holder._release_resume_checkpoint_consumer(active_snapshot)

    assert not active_snapshot.exists()


def _crash_with_shared_resume_consumer_worker(checkpoint, ready_file):
    entry = _entry_spy()
    snapshot, identity = entry._freeze_resume_checkpoint(Path(checkpoint))
    entry._resolved_resume_identity = identity
    entry._acquire_resume_checkpoint_consumer(snapshot)
    entry._complete_resume_checkpoint_consumer_acquisition(snapshot)
    Path(ready_file).write_text(str(snapshot), encoding="utf-8")
    os._exit(0)


def test_resume_consumer_os_exit_is_reclaimed_by_next_freeze(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    ready = tmp_path / "consumer-crash-ready.txt"
    process = mp.get_context("spawn").Process(
        target=_crash_with_shared_resume_consumer_worker,
        args=(str(checkpoint), str(ready)),
    )
    process.start()
    process.join(_GLOO_PREVIEW_TIMEOUT)
    if process.is_alive():
        process.terminate()
        raise TimeoutError("shared consumer crash worker did not stop")
    assert process.exitcode == 0
    stale_snapshot = Path(ready.read_text(encoding="utf-8"))
    assert stale_snapshot.exists()

    entry = _entry_spy()
    replacement_snapshot, _identity = entry._freeze_resume_checkpoint(checkpoint)
    try:
        assert not stale_snapshot.exists()
    finally:
        entry._cleanup_resume_checkpoint_snapshot(replacement_snapshot)


def _spawn_gloo_stage_failure(tmp_path, worker, checkpoint, result_prefix):
    context = mp.spawn(
        worker,
        args=(
            2,
            str(tmp_path / f"{result_prefix}_init"),
            str(checkpoint),
            str(tmp_path),
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
    while not context.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError(f"Gloo {result_prefix} failure test timed out")
    return tuple(
        ast.literal_eval(
            (tmp_path / f"{result_prefix}-rank{rank}.txt").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_sampler_stage_aggregates_rank_zero_failure_before_cleanup(tmp_path):
    checkpoint = tmp_path / "sampler-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)

    first, second = _spawn_gloo_stage_failure(
        tmp_path, _gloo_sampler_stage_failure_worker, checkpoint, "sampler"
    )

    assert first == second
    assert first[:2] == ("error", "RuntimeError")
    assert "sampler preparation failed" in first[2]
    assert "rank 0: RuntimeError: injected sampler rank 0 failure" in first[2]
    assert not list(tmp_path.glob(".sampler-latest_model.tar.resume-*.snapshot"))


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_sampler_stage_aggregates_rank_one_failure_before_cleanup(tmp_path):
    checkpoint = tmp_path / "sampler-rank1-latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)

    first, second = _spawn_gloo_stage_failure(
        tmp_path, _gloo_sampler_stage_failure_worker, checkpoint, "sampler"
    )

    assert first == second
    assert first[:2] == ("error", "RuntimeError")
    assert "sampler preparation failed" in first[2]
    assert "rank 1: RuntimeError: injected sampler rank 1 failure" in first[2]
    assert not list(tmp_path.glob(".sampler-rank1-latest_model.tar.resume-*.snapshot"))


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_gloo_original_trainer_controls_resume_hooks_and_nested_aliases(tmp_path):
    exp_path = tmp_path / "original-lifecycle-exp"
    checkpoint_dir = exp_path / "checkpoints" / "persistent-run"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "latest_model.tar"
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    torch.save(
        {
            "epoch": 7,
            "batch_cursor": 1,
            "n_steps": 3,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint,
    )

    mp.spawn(
        _gloo_original_trainer_lifecycle_worker,
        args=(
            2,
            str(tmp_path / "original_lifecycle_init"),
            str(checkpoint),
            str(tmp_path),
        ),
        nprocs=2,
        join=True,
    )

    loaded_snapshots = []
    for rank in range(2):
        outcome = ast.literal_eval(
            (tmp_path / f"original-lifecycle-rank{rank}.txt").read_text(
                encoding="utf-8"
            )
        )
        assert outcome[:5] == (True, True, True, (7, 3, 2), True)
        assert Path(outcome[5]) == checkpoint_dir
        assert Path(outcome[6]) == exp_path / "logs" / "persistent-run"
        assert len(outcome[7]) == 1
        loaded_snapshot = Path(outcome[7][0])
        assert loaded_snapshot.parent == (
            exp_path / "resume-staging" / f"rank-{rank}"
        )
        assert Path(outcome[8]) == loaded_snapshot
        assert Path(outcome[9]) == loaded_snapshot
        loaded_snapshots.append(loaded_snapshot)
    assert loaded_snapshots[0] != loaded_snapshots[1]
    assert all(not snapshot.exists() for snapshot in loaded_snapshots)
    assert safe_torch_load(checkpoint)["epoch"] == 8
    assert (checkpoint_dir / "model_0008.pth").exists()
    assert _entry_spy()._resume_checkpoint_path(
        {"trainer": {"exp_path": str(exp_path)}},
        Namespace(resume=True, resume_from=None),
    ) == checkpoint


def _hold_resume_snapshot_worker(checkpoint, ready_file, release_file):
    entry = _entry_spy()
    snapshot, _identity = entry._freeze_resume_checkpoint(Path(checkpoint))
    Path(ready_file).write_text(str(snapshot), encoding="utf-8")
    deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
    while not Path(release_file).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("snapshot holder timed out")
        time.sleep(0.02)
    entry._cleanup_resume_checkpoint_snapshot(snapshot)


def _crash_with_resume_snapshot_worker(checkpoint, ready_file):
    snapshot, _identity = _entry_spy()._freeze_resume_checkpoint(Path(checkpoint))
    Path(ready_file).write_text(str(snapshot), encoding="utf-8")
    os._exit(0)


def test_resume_snapshot_scan_preserves_active_lease_and_reclaims_stale(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    ready = tmp_path / "holder-ready.txt"
    release = tmp_path / "holder-release.txt"
    holder = mp.get_context("spawn").Process(
        target=_hold_resume_snapshot_worker,
        args=(str(checkpoint), str(ready), str(release)),
    )
    holder.start()
    deadline = time.monotonic() + _GLOO_PREVIEW_TIMEOUT
    while not ready.exists():
        if time.monotonic() >= deadline:
            holder.terminate()
            raise TimeoutError("snapshot holder did not become ready")
        time.sleep(0.02)
    active_snapshot = Path(ready.read_text(encoding="utf-8"))
    stale_token = "123-" + "a" * 32
    stale_snapshot = checkpoint.with_name(
        f".{checkpoint.name}.resume-{stale_token}.snapshot"
    )
    os.link(checkpoint, stale_snapshot)
    other_checkpoint = tmp_path / "other-latest_model.tar"
    torch.save({"epoch": 9, "n_steps": 100, "batch_cursor": 1}, other_checkpoint)
    unrelated_snapshot = other_checkpoint.with_name(
        f".{other_checkpoint.name}.resume-{stale_token}.snapshot"
    )
    os.link(other_checkpoint, unrelated_snapshot)

    entry = _entry_spy()
    concurrent_snapshot, _identity = entry._freeze_resume_checkpoint(checkpoint)
    try:
        assert active_snapshot.exists()
        assert not stale_snapshot.exists()
        assert unrelated_snapshot.exists()
    finally:
        entry._cleanup_resume_checkpoint_snapshot(concurrent_snapshot)
        release.touch()
        holder.join(_GLOO_PREVIEW_TIMEOUT)
        if holder.is_alive():
            holder.terminate()
            raise TimeoutError("snapshot holder did not stop")
        unrelated_snapshot.unlink(missing_ok=True)
    assert holder.exitcode == 0
    assert not active_snapshot.exists()


def test_resume_snapshot_scan_reclaims_lease_left_by_abrupt_exit(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    ready = tmp_path / "crash-ready.txt"
    process = mp.get_context("spawn").Process(
        target=_crash_with_resume_snapshot_worker,
        args=(str(checkpoint), str(ready)),
    )
    process.start()
    process.join(_GLOO_PREVIEW_TIMEOUT)
    if process.is_alive():
        process.terminate()
        raise TimeoutError("snapshot crash worker did not stop")
    assert process.exitcode == 0
    stale_snapshot = Path(ready.read_text(encoding="utf-8"))
    assert stale_snapshot.exists()

    entry = _entry_spy()
    replacement_snapshot, _identity = entry._freeze_resume_checkpoint(checkpoint)
    try:
        assert not stale_snapshot.exists()
        assert replacement_snapshot.exists()
    finally:
        entry._cleanup_resume_checkpoint_snapshot(replacement_snapshot)


def test_next_freeze_reclaims_only_strict_canonical_stale_snapshot_temp(tmp_path):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    token = "123-" + "a" * 32
    stale_temp = tmp_path / f".{checkpoint.name}.resume-{token}.snapshot.tmp"
    stale_temp.write_bytes(b"abandoned-copy")
    unrelated_basename = tmp_path / f".other-{checkpoint.name}.resume-{token}.snapshot.tmp"
    unrelated_basename.write_bytes(b"unrelated")
    malformed_suffix = tmp_path / (
        f".{checkpoint.name}.resume-{token}.snapshot.tmp.extra"
    )
    malformed_suffix.write_bytes(b"unrelated")

    entry = _entry_spy()
    replacement_snapshot, _identity = entry._freeze_resume_checkpoint(checkpoint)
    try:
        assert not stale_temp.exists()
        assert unrelated_basename.exists()
        assert malformed_suffix.exists()
        assert replacement_snapshot.exists()
    finally:
        entry._cleanup_resume_checkpoint_snapshot(replacement_snapshot)


def test_stale_temp_reclaim_treats_checkpoint_basename_literally(tmp_path):
    checkpoint = tmp_path / "foo?.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)
    token = "123-" + "a" * 32
    other_temp = tmp_path / f".fooa.tar.resume-{token}.snapshot.tmp"
    other_temp.write_bytes(b"active-other-writer")

    entry = _entry_spy()
    snapshot, _identity = entry._freeze_resume_checkpoint(checkpoint)
    try:
        assert other_temp.exists()
        assert snapshot.exists()
    finally:
        entry._cleanup_resume_checkpoint_snapshot(snapshot)


@pytest.mark.parametrize("failure_point", ["lock", "register", "publish"])
def test_freeze_resume_checkpoint_rolls_back_every_post_copy_failure(
    monkeypatch, tmp_path, failure_point
):
    checkpoint = tmp_path / "latest_model.tar"
    torch.save({"epoch": 7, "n_steps": 100, "batch_cursor": 1}, checkpoint)

    if failure_point == "lock":
        import fcntl

        original_flock = fcntl.flock
        exclusive_calls = 0

        def fail_lease_lock(fd, operation):
            nonlocal exclusive_calls
            if operation & fcntl.LOCK_EX:
                exclusive_calls += 1
                if exclusive_calls == 2:
                    raise OSError("injected lease lock failure")
            return original_flock(fd, operation)

        monkeypatch.setattr("core.trainer.entry.fcntl.flock", fail_lease_lock)
    elif failure_point == "register":
        def fail_register(*_args, **_kwargs):
            raise RuntimeError("injected register failure")

        monkeypatch.setattr("core.trainer.entry.atexit.register", fail_register)
    else:
        original_replace = os.replace

        def fail_snapshot_publish(source, target):
            if ".resume-" in Path(target).name:
                raise OSError("injected snapshot publish failure")
            return original_replace(source, target)

        monkeypatch.setattr("core.trainer.entry.os.replace", fail_snapshot_publish)

    expected_error = {
        "lock": "injected lease lock failure",
        "register": "injected register failure",
        "publish": "injected snapshot publish failure",
    }[failure_point]
    with pytest.raises(Exception, match=expected_error):
        _entry_spy()._freeze_resume_checkpoint(checkpoint)

    assert not list(tmp_path.glob(".latest_model.tar.resume-*"))


@pytest.mark.unit
def test_layer_f_runs_in_cpu_path_and_respects_global_switch():
    import numpy as np

    from core.data.post_augment import apply_post_augmentations
    from core.data.simulation_configs import SimulationConfigs

    cfg = SimulationConfigs()
    cfg.noisy_post_mix_degradations_enabled = True
    cfg.prob_dc_offset = 1.0
    cfg.dc_offset_range = (0.1, 0.1)
    cfg.prob_clip = 0.0
    noisy = np.zeros(16, dtype=np.float32)
    operations, facts = [], {}
    output = apply_post_augmentations(
        noisy,
        16_000,
        cfg,
        rng=np.random.default_rng(0),
        applied_ops_out=operations,
        clipping_facts_out=facts,
    )
    assert operations == ["dc_offset"]
    assert not np.array_equal(output, noisy)
    assert facts == {}

    cfg.noisy_post_mix_degradations_enabled = False
    operations.clear()
    skipped = apply_post_augmentations(
        noisy,
        16_000,
        cfg,
        rng=np.random.default_rng(0),
        applied_ops_out=operations,
        clipping_facts_out=facts,
    )
    np.testing.assert_array_equal(skipped, noisy)
    assert operations == []


pytestmark = [
    pytest.mark.unit,
    pytest.mark.usefixtures("preserve_tensorboard_module_state"),
]
