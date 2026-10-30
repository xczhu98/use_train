"""BaseTrainer checkpoint 完整性与 DDP 保存同步契约测试。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import core.trainer.base_trainer as base_trainer_module
from core.trainer.base_trainer import BaseTrainer


class _TinyModel(nn.Module):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)


class _ExtraStateModel(_TinyModel):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_extra_state = None

    def get_extra_state(self):
        return {"schema_version": 1}

    def set_extra_state(self, state) -> None:
        self.loaded_extra_state = state


def _bare_trainer(tmp_path: Path, *, rank: int = 0, world_size: int = 1) -> BaseTrainer:
    """绕过 TensorBoard 初始化，构造 checkpoint helper 所需的最小 trainer。"""
    model = _TinyModel()
    trainer = object.__new__(BaseTrainer)
    trainer.rank = rank
    trainer.world_size = world_size
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.models = {"default": model}
    trainer._is_multi_model = False
    trainer.ckpt_dir = tmp_path
    trainer.accumulation_steps = 1
    return trainer


def _resume_trainer(tmp_path: Path) -> BaseTrainer:
    """构造可执行 resume 状态恢复所需的最小单优化器 trainer。"""
    trainer = _bare_trainer(tmp_path)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1, momentum=0.9)
    trainer.optimizers = {"default": trainer.optimizer}
    trainer._is_multi_optimizer = False
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
    trainer.schedulers = {"default": trainer.scheduler}
    trainer._is_multi_scheduler = False
    trainer.use_amp = False
    trainer.amp_dtype = torch.float16
    trainer.ema = None
    trainer.train_sampler = None
    trainer.config = {}
    return trainer


def _write_resume_checkpoint(trainer: BaseTrainer, path: Path) -> None:
    loss = trainer.model.linear.weight.sum()
    loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    torch.save(
        {
            "epoch": 2,
            "n_steps": 4,
            "model": {key: value.clone() for key, value in trainer.model.state_dict().items()},
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
        },
        path,
    )


def _build_run_dir_trainer(
    exp_path: Path,
    *,
    config_marker: str = "fresh",
    resume: bool = False,
    resume_from: Path | None = None,
    resume_load_from: Path | None = None,
    resume_source_from: Path | None = None,
) -> BaseTrainer:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return BaseTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.MSELoss(),
        config={
            "trainer": {"exp_path": str(exp_path)},
            "meta": {},
            "config_marker": config_marker,
        },
        resume=resume,
        resume_from=str(resume_from) if resume_from is not None else None,
        resume_load_from=(
            str(resume_load_from) if resume_load_from is not None else None
        ),
        resume_source_from=(
            str(resume_source_from) if resume_source_from is not None else None
        ),
    )


def _close_run_dir_trainer(trainer: BaseTrainer) -> None:
    if trainer.writer is not None:
        trainer.writer.close()
    root = logging.getLogger()
    log_root = str(trainer.log_dir.resolve())
    for handler in list(root.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if str(Path(handler.baseFilename).resolve()).startswith(log_root):
            handler.close()
            root.removeHandler(handler)


def test_same_second_constructions_claim_distinct_run_directories(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        base_trainer_module.time,
        "strftime",
        lambda *_args, **_kwargs: "2026-08-24_12-34-56",
    )
    monkeypatch.setattr(BaseTrainer, "_save_experiment_context", lambda self: None)

    first = _build_run_dir_trainer(tmp_path)
    second = _build_run_dir_trainer(tmp_path)
    try:
        assert first.ckpt_dir.name == "2026-08-24_12-34-56"
        assert second.ckpt_dir.name == (
            f"2026-08-24_12-34-56_p{base_trainer_module.os.getpid()}"
        )
        assert first.ckpt_dir != second.ckpt_dir
    finally:
        _close_run_dir_trainer(first)
        _close_run_dir_trainer(second)


def test_resumed_checkpoint_binds_content_addressed_resolved_config(
    tmp_path: Path,
) -> None:
    fresh = _build_run_dir_trainer(tmp_path, config_marker="fresh")
    try:
        fresh._save_checkpoint(epoch=1)
        checkpoint = fresh.ckpt_dir / "latest_model.tar"
        original_sidecar = fresh.ckpt_dir / "resolved_config.yaml"
        original_text = original_sidecar.read_text(encoding="utf-8")
    finally:
        _close_run_dir_trainer(fresh)

    resumed = _build_run_dir_trainer(
        tmp_path,
        config_marker="resumed",
        resume=True,
        resume_from=checkpoint,
    )
    try:
        resumed._save_checkpoint(epoch=2)
        state = resumed._safe_load(checkpoint)
        provenance = state["resolved_config_provenance"]
        resumed_sidecar = resumed.ckpt_dir / provenance["source"]

        assert original_sidecar.read_text(encoding="utf-8") == original_text
        assert provenance["source"].startswith("resolved_config_resumed_")
        assert resumed_sidecar.is_file()
        assert "config_marker: resumed" in resumed_sidecar.read_text(encoding="utf-8")
        assert state["resolved_config"]["config_marker"] == "resumed"
    finally:
        _close_run_dir_trainer(resumed)


@pytest.mark.unit
def test_resume_load_from_reads_snapshot_but_keeps_persistent_run_directory(
    tmp_path: Path, monkeypatch
) -> None:
    exp_path = tmp_path / "exp"
    checkpoint_dir = exp_path / "checkpoints" / "persistent-run"
    checkpoint_dir.mkdir(parents=True)
    source = checkpoint_dir / "latest_model.tar"
    snapshot = tmp_path / "resume-staging" / "rank-0.snapshot"
    snapshot.parent.mkdir()

    source_trainer = _build_run_dir_trainer(exp_path)
    try:
        source_trainer.model.linear.weight.sum().backward()
        source_trainer.optimizer.step()
        source_trainer.scheduler.step()
        expected_model = {
            key: value.detach().clone()
            for key, value in source_trainer.model.state_dict().items()
        }
        expected_lr = source_trainer.optimizer.param_groups[0]["lr"]
        expected_scheduler_epoch = source_trainer.scheduler.last_epoch
        source_trainer._save_checkpoint(epoch=2)
        source.write_bytes((source_trainer.ckpt_dir / "latest_model.tar").read_bytes())
    finally:
        _close_run_dir_trainer(source_trainer)
    snapshot.write_bytes(source.read_bytes())

    loaded_paths: list[Path] = []
    original_safe_load = BaseTrainer._safe_load

    def record_safe_load(path, **kwargs):
        loaded_paths.append(Path(path))
        return original_safe_load(path, **kwargs)

    monkeypatch.setattr(BaseTrainer, "_safe_load", staticmethod(record_safe_load))
    resumed = _build_run_dir_trainer(
        exp_path,
        config_marker="resumed",
        resume=True,
        resume_from=snapshot,
        resume_load_from=snapshot,
        resume_source_from=source,
    )
    try:
        assert resumed._resume_from == str(snapshot)
        assert resumed._resume_load_from == str(snapshot)
        assert resumed.ckpt_dir == checkpoint_dir
        assert resumed.log_dir == exp_path / "logs" / "persistent-run"
        assert loaded_paths == [snapshot]
        assert resumed.start_epoch == 3
        assert resumed.n_steps == 0
        for key, value in expected_model.items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value)
        assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
        assert resumed.scheduler.last_epoch == expected_scheduler_epoch

        resumed._save_checkpoint(epoch=3)
        assert source.exists()
        assert (checkpoint_dir / "model_0003.pth").is_file()
    finally:
        _close_run_dir_trainer(resumed)


@pytest.mark.unit
def test_custom_resume_override_only_observes_immutable_snapshot(tmp_path: Path) -> None:
    exp_path = tmp_path / "exp"
    source = exp_path / "checkpoints" / "persistent-run" / "latest_model.tar"
    source.parent.mkdir(parents=True)
    snapshot = tmp_path / "resume-staging" / "rank-0.snapshot"
    snapshot.parent.mkdir()
    torch.save({"epoch": 2}, source)
    snapshot.write_bytes(source.read_bytes())
    observed = {}

    class OverrideTrainer(BaseTrainer):
        def _resume_checkpoint(self):
            observed["resume_from"] = Path(self._resume_from)
            observed["resume_load_from"] = Path(self._resume_load_from)
            observed["has_source_attr"] = hasattr(self, "_resume_source_from")
            self._after_resume_checkpoint({"epoch": 2})

        def _after_resume_checkpoint(self, checkpoint_state):
            observed["after_resume_epoch"] = checkpoint_state["epoch"]

    model = _TinyModel()
    trainer = OverrideTrainer(
        rank=0,
        world_size=1,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=torch.optim.lr_scheduler.StepLR(
            torch.optim.SGD(model.parameters(), lr=0.1), step_size=1
        ),
        loss_fn=nn.MSELoss(),
        config={"trainer": {"exp_path": str(exp_path)}, "meta": {}},
        resume=True,
        resume_from=str(snapshot),
        resume_load_from=str(snapshot),
        resume_source_from=str(source),
    )
    try:
        assert observed == {
            "resume_from": snapshot,
            "resume_load_from": snapshot,
            "has_source_attr": False,
            "after_resume_epoch": 2,
        }
        assert trainer.ckpt_dir == source.parent
    finally:
        _close_run_dir_trainer(trainer)


def test_resolved_config_sidecar_reuses_matching_identity(tmp_path: Path) -> None:
    trainer = _build_run_dir_trainer(tmp_path, config_marker="same")
    try:
        original = trainer._resolved_config_provenance
        trainer._save_experiment_context()

        assert trainer._resolved_config_provenance == original
        assert sorted(path.name for path in trainer.ckpt_dir.glob("resolved_config*")) == [
            "resolved_config.yaml"
        ]
    finally:
        _close_run_dir_trainer(trainer)


def test_resolved_config_sidecar_rejects_content_address_collision(
    tmp_path: Path,
) -> None:
    from core.trainer.checkpoint_state import resolved_config_identity

    trainer = _build_run_dir_trainer(tmp_path, config_marker="fresh")
    try:
        trainer.config = {
            **trainer.config,
            "config_marker": "resumed",
        }
        identity = resolved_config_identity(trainer.config)
        collision = trainer.ckpt_dir / f"resolved_config_resumed_{identity}.yaml"
        collision.write_text("config_marker: tampered\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="identity collision"):
            trainer._save_experiment_context()
    finally:
        _close_run_dir_trainer(trainer)


def test_run_dir_claim_skips_existing_pid_fallback(tmp_path: Path, monkeypatch) -> None:
    timestamp = "2026-08-24_12-34-56"
    pid = 24680
    checkpoint_base = tmp_path / "checkpoints"
    (checkpoint_base / timestamp).mkdir(parents=True)
    (checkpoint_base / f"{timestamp}_p{pid}").mkdir()
    monkeypatch.setattr(base_trainer_module.time, "strftime", lambda *_args: timestamp)
    monkeypatch.setattr(base_trainer_module.os, "getpid", lambda: pid)

    claimed = BaseTrainer._claim_checkpoint_run_dir(checkpoint_base)

    assert claimed.name == f"{timestamp}_p{pid}_1"
    assert claimed.is_dir()


def test_ddp_run_dir_claim_is_rank0_only_and_broadcasts_same_path(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_base = tmp_path / "checkpoints"
    broadcasts: list[tuple[int, str | None]] = []
    shared_outcome = None
    claim_ranks: list[int] = []
    active_rank = 0
    original_claim = BaseTrainer._claim_checkpoint_run_dir

    def claim(base: Path) -> Path:
        claim_ranks.append(active_rank)
        return original_claim(base)

    def broadcast_object_list(outcome, src: int) -> None:
        nonlocal shared_outcome
        broadcasts.append((active_rank, outcome[0][0] if outcome[0] else None))
        if active_rank == src:
            shared_outcome = outcome[0]
        else:
            outcome[0] = shared_outcome

    monkeypatch.setattr(BaseTrainer, "_claim_checkpoint_run_dir", staticmethod(claim))
    monkeypatch.setattr(base_trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(base_trainer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        base_trainer_module.dist,
        "broadcast_object_list",
        broadcast_object_list,
    )

    rank0_path, rank0_claimed = BaseTrainer._select_or_allocate_checkpoint_run_dir(
        checkpoint_base, rank=0, world_size=2
    )
    active_rank = 1
    rank1_path, rank1_claimed = BaseTrainer._select_or_allocate_checkpoint_run_dir(
        checkpoint_base, rank=1, world_size=2
    )

    assert rank0_path == rank1_path
    assert rank0_claimed is rank1_claimed is True
    assert rank0_path.is_dir()
    assert claim_ranks == [0]
    assert broadcasts == [(0, "ok"), (1, None)]


def test_checkpoint_loaders_separate_resume_and_model_only_safety(
    tmp_path: Path, monkeypatch
) -> None:
    import core.utils.safe_load as safe_load_module

    trainer = _bare_trainer(tmp_path)
    trainer.ema = None
    checkpoint = tmp_path / "checkpoint.pt"
    calls: list[tuple[Path, dict]] = []

    def fake_safe_load(path, **kwargs):
        calls.append((Path(path), kwargs))
        return {"model": trainer.model.state_dict()}

    monkeypatch.setattr(safe_load_module, "safe_torch_load", fake_safe_load)
    monkeypatch.setattr(safe_load_module, "safe_model_weights_load", fake_safe_load)

    BaseTrainer._safe_load(checkpoint)
    trainer._load_pretrained_local(checkpoint, None)

    assert calls == [
        (
            checkpoint,
            {
                "allow_unsafe": True,
                "purpose": "resume (optimizer state)",
            },
        ),
        (
            checkpoint,
            {
                "purpose": "initialize trainer model weights from pretrained checkpoint",
            },
        ),
    ]


def test_model_only_loader_accepts_safetensors_with_strict_state_validation(
    tmp_path: Path,
) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    trainer = _bare_trainer(tmp_path)
    trainer.ema = None
    checkpoint = tmp_path / "model.safetensors"
    safetensors.save_file(trainer.model.state_dict(), str(checkpoint))

    summary = trainer._load_pretrained_local(checkpoint, None)

    assert summary[0] is None
    assert summary[1] == len(trainer.model.state_dict())


def test_resume_progress_bounds_are_optional_without_sized_train_loader(
    tmp_path: Path,
) -> None:
    """直接 BaseTrainer 或 iterable-only loader 仍保留原 resume 合同。"""
    for train_loader in (None, iter([object()])):
        trainer = _resume_trainer(tmp_path)
        if train_loader is not None:
            trainer.train_loader = train_loader
        checkpoint = {
            "epoch": 2,
            "n_steps": 4,
            "batch_cursor": 8,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
        }

        trainer._validate_resume_checkpoint(checkpoint)


def test_resume_progress_uses_sized_train_loader_bound(tmp_path: Path) -> None:
    trainer = _resume_trainer(tmp_path)
    trainer.train_loader = [object()] * 4
    checkpoint = {
        "epoch": 2,
        "n_steps": 4,
        "batch_cursor": 5,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
    }

    with pytest.raises(ValueError, match="train loader length 4"):
        trainer._validate_resume_checkpoint(checkpoint)


def test_resume_rejects_missing_optimizer_or_scheduler_state(tmp_path: Path) -> None:
    for missing_key, state_name in (
        ("optimizer", "optimizer"),
        ("scheduler", "scheduler"),
    ):
        trainer = _resume_trainer(tmp_path / missing_key)
        checkpoint = tmp_path / f"missing_{missing_key}.tar"
        _write_resume_checkpoint(trainer, checkpoint)
        state = trainer._safe_load(checkpoint)
        state.pop(missing_key)
        torch.save(state, checkpoint)

        with pytest.raises(
            RuntimeError,
            match=f"missing required {state_name} state '{state_name}'",
        ):
            trainer._load_resume_checkpoint_local(checkpoint)


def test_fp16_resume_requires_and_restores_scaler_state(tmp_path: Path) -> None:
    trainer = _resume_trainer(tmp_path)
    trainer.use_amp = True
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    checkpoint = tmp_path / "fp16_resume.tar"
    _write_resume_checkpoint(trainer, checkpoint)

    with pytest.raises(RuntimeError, match="missing required AMP scaler state 'scaler'"):
        trainer._load_resume_checkpoint_local(checkpoint)

    state = trainer._safe_load(checkpoint)
    source_scaler = torch.amp.GradScaler("cpu", init_scale=2048.0)
    state["scaler"] = source_scaler.state_dict()
    torch.save(state, checkpoint)

    trainer._load_resume_checkpoint_local(checkpoint)

    assert trainer.scaler.get_scale() == 2048.0


@pytest.mark.unit
def test_resume_snapshot_moves_nested_tensors_to_cpu(tmp_path: Path) -> None:
    trainer = _resume_trainer(tmp_path)
    snapshot = trainer._snapshot_resume_state()

    def assert_cpu(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
        elif isinstance(value, dict):
            for nested in value.values():
                assert_cpu(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                assert_cpu(nested)

    assert_cpu(snapshot)


@pytest.mark.unit
def test_negative_max_steps_is_rejected_during_initialization(tmp_path: Path) -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = {
        "trainer": {"exp_path": str(tmp_path), "max_steps": -1},
        "meta": {},
    }

    with pytest.raises(ValueError, match="max_steps must be non-negative"):
        BaseTrainer(
            rank=0,
            world_size=1,
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
            loss_fn=nn.MSELoss(),
            config=config,
        )


@pytest.mark.parametrize("value", [-0.5, 1.9, True])
def test_max_steps_rejects_non_integral_values(value) -> None:
    with pytest.raises(ValueError):
        base_trainer_module._normalize_max_steps(value)


def test_max_steps_accepts_integer_string() -> None:
    assert base_trainer_module._normalize_max_steps("3") == 3


def test_strict_model_state_accepts_expected_extra_state(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)
    model = _ExtraStateModel()
    trainer.model = model
    trainer.models = {"default": model}
    state = model.state_dict()

    assert state["_extra_state"] == {"schema_version": 1}
    trainer._load_model_state_checked(state, context="contract")

    assert model.loaded_extra_state == {"schema_version": 1}


def test_strict_model_state_accepts_legacy_module_prefix(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)
    expected = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    prefixed = {f"module.{key}": value for key, value in expected.items()}

    trainer._load_model_state_checked(prefixed, context="contract")

    for key, value in expected.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: state.pop("linear.bias"), "missing"),
        (lambda state: state.__setitem__("extra", torch.zeros(1)), "unexpected"),
        (lambda state: state.__setitem__("linear.weight", torch.zeros(1)), "shape mismatch"),
        (lambda state: state.__setitem__("linear.weight", "not a tensor"), "non-tensor"),
    ],
)
def test_strict_model_state_rejects_incomplete_or_invalid_state(
    tmp_path: Path, mutate, message: str
) -> None:
    trainer = _bare_trainer(tmp_path)
    state = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    mutate(state)

    with pytest.raises(RuntimeError, match=message):
        trainer._load_model_state_checked(state, context="contract")


def test_strict_model_state_rejects_zero_match(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)

    with pytest.raises(RuntimeError, match="zero model parameters matched"):
        trainer._load_model_state_checked(
            {"other.weight": torch.zeros(2, 2)}, context="contract"
        )


@pytest.fixture
def fake_distributed(monkeypatch):
    """在单进程内记录保存同步顺序，不初始化真实 process group。"""
    calls = {"all_reduce": 0, "all_gather_object": 0, "barrier": 0}

    monkeypatch.setattr(base_trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(base_trainer_module.dist, "is_initialized", lambda: True)

    def all_reduce(_tensor, op=None):
        calls["all_reduce"] += 1

    def all_gather_object(output, value):
        calls["all_gather_object"] += 1
        for index in range(len(output)):
            output[index] = value

    def barrier():
        calls["barrier"] += 1

    monkeypatch.setattr(base_trainer_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(base_trainer_module.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(base_trainer_module.dist, "barrier", barrier)
    return calls


@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_sync_collective_sequence_matches_on_both_ranks(
    tmp_path: Path, monkeypatch, rank: int
) -> None:
    trainer = _bare_trainer(tmp_path, rank=rank, world_size=2)
    sequence = []
    monkeypatch.setattr(base_trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(base_trainer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        base_trainer_module.dist,
        "all_reduce",
        lambda *_args, **_kwargs: sequence.append("all_reduce"),
    )
    monkeypatch.setattr(
        base_trainer_module.dist,
        "barrier",
        lambda: sequence.append("barrier"),
    )

    trainer._sync_checkpoint_save_result(
        None, barrier_phase="barrier_after_checkpoint_save"
    )

    assert sequence == ["all_reduce", "barrier"]


@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_failure_collective_sequence_matches_on_both_ranks(
    tmp_path: Path, monkeypatch, rank: int
) -> None:
    trainer = _bare_trainer(tmp_path, rank=rank, world_size=2)
    sequence = []
    monkeypatch.setattr(base_trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(base_trainer_module.dist, "is_initialized", lambda: True)

    def all_reduce(status, **_kwargs):
        sequence.append("all_reduce")
        status.zero_()

    def all_gather_object(output, _value):
        sequence.append("all_gather_object")
        output[:] = ["rank 0 disk full", None]

    monkeypatch.setattr(base_trainer_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(
        base_trainer_module.dist, "all_gather_object", all_gather_object
    )
    monkeypatch.setattr(
        base_trainer_module.dist,
        "barrier",
        lambda: sequence.append("barrier"),
    )

    local_error = "rank 0 disk full" if rank == 0 else None
    with pytest.raises(RuntimeError, match="rank 0 disk full"):
        trainer._sync_checkpoint_save_result(
            local_error, barrier_phase="barrier_after_checkpoint_save"
        )

    assert sequence == ["all_reduce", "all_gather_object"]


def test_save_checkpoint_syncs_success_and_skips_barrier_on_rank0_failure(
    tmp_path: Path, fake_distributed
) -> None:
    trainer = _bare_trainer(tmp_path, rank=0, world_size=2)
    trainer._build_state = lambda epoch, **_kwargs: {"epoch": epoch}
    saved_paths: list[Path] = []
    trainer._atomic_save_aliases = (
        lambda _state, paths, **_kwargs: saved_paths.extend(paths)
    )

    trainer._save_checkpoint(epoch=3)

    assert [path.name for path in saved_paths] == ["latest_model.tar", "model_0003.pth"]
    assert fake_distributed == {
        "all_reduce": 1,
        "all_gather_object": 1,
        "barrier": 1,
    }

    failed = _bare_trainer(tmp_path / "failed", rank=0, world_size=2)
    failed._build_state = lambda epoch, **_kwargs: {"epoch": epoch}
    failed._atomic_save_aliases = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(OSError("disk full"))

    with pytest.raises(RuntimeError, match="disk full"):
        failed._save_checkpoint(epoch=4)

    assert fake_distributed == {
        "all_reduce": 2,
        "all_gather_object": 3,
        "barrier": 1,
    }


def test_train_deduplicates_epoch_best_and_periodic_checkpoint(tmp_path: Path) -> None:
    """save_interval=1 时改善验证只构造一次，且状态在 scheduler step 后。"""
    trainer, scheduler, built_states, saved_paths = _checkpoint_loop_trainer(
        tmp_path,
        epochs=1,
        save_interval=1,
        validation_epoch_interval=1,
        val_losses=[0.5],
    )

    trainer.train()

    assert built_states == [(1, 1)]
    assert scheduler.last_epoch == 1
    assert [path.name for path in saved_paths] == [
        "latest_model.tar", "model_0001.pth", "best_model.tar"
    ]


def _checkpoint_loop_trainer(
    tmp_path: Path,
    *,
    epochs: int,
    save_interval: int,
    validation_epoch_interval: int,
    val_losses: list[float],
    patience: int = 0,
) -> tuple[BaseTrainer, object, list[tuple[int, int]], list[Path]]:
    """构造可观察 scheduler 与 checkpoint 时序的最小训练循环。"""
    trainer = _bare_trainer(tmp_path)
    trainer.start_epoch = 1
    trainer.epochs = epochs
    trainer.current_epoch = 1
    trainer.max_steps = None
    trainer.n_steps = 0
    trainer.train_sampler = None
    trainer.train_dataset = None
    trainer.validation_epoch_interval = validation_epoch_interval
    trainer.validation_steps_interval = 0
    trainer.save_top_k = 0
    trainer.save_interval = save_interval
    trainer.clear_cache_each_epoch = False
    trainer.lr_per_step = False
    trainer._is_multi_scheduler = False
    trainer.best_score = float("inf")
    trainer.best_epoch = -1
    trainer.best_at_step = None
    trainer._patience_counter = 0
    trainer.patience = patience
    trainer.train_losses = []
    trainer.val_losses = []
    trainer.curve_epochs = []
    trainer._train_epoch = lambda _epoch, step_callback=None: 1.0
    values = iter(val_losses)
    trainer._validation_epoch = lambda _epoch: next(values)
    trainer._log_scalar = lambda *_args, **_kwargs: None
    trainer._on_loss_curve_point_added = lambda *_args: None
    trainer._should_save_curve = lambda _count: False

    class _Scheduler:
        def __init__(self) -> None:
            self.last_epoch = 0

        def step(self) -> None:
            self.last_epoch += 1

        def state_dict(self) -> dict[str, int]:
            return {"last_epoch": self.last_epoch}

    scheduler = _Scheduler()
    trainer.scheduler = scheduler
    built_states: list[tuple[int, int]] = []
    saved_paths: list[Path] = []

    def build_state(epoch: int, **_kwargs) -> dict[str, object]:
        built_states.append((epoch, scheduler.last_epoch))
        return {"epoch": epoch, "scheduler": scheduler.state_dict()}

    trainer._build_state = build_state
    trainer._atomic_save_aliases = (
        lambda _state, paths, **_kwargs: saved_paths.extend(paths)
    )
    return trainer, scheduler, built_states, saved_paths


def test_periodic_boundary_checkpoint_uses_post_scheduler_state(
    tmp_path: Path,
) -> None:
    """非周期 best 与后续周期保存均使用各自 epoch scheduler step 后的状态。"""
    trainer, scheduler, built_states, saved_paths = _checkpoint_loop_trainer(
        tmp_path,
        epochs=2,
        save_interval=2,
        validation_epoch_interval=1,
        val_losses=[0.5, 0.6],
    )

    trainer.train()

    assert built_states == [(1, 1), (2, 2)]
    assert scheduler.last_epoch == 2
    assert [path.name for path in saved_paths] == [
        "latest_model.tar",
        "model_0001.pth",
        "best_model.tar",
        "latest_model.tar",
        "model_0002.pth",
    ]


def test_final_fallback_without_validation_uses_post_scheduler_state(
    tmp_path: Path,
) -> None:
    """关闭验证和周期保存时，最终兜底仍保存 scheduler step 后状态。"""
    trainer, scheduler, built_states, saved_paths = _checkpoint_loop_trainer(
        tmp_path,
        epochs=2,
        save_interval=0,
        validation_epoch_interval=0,
        val_losses=[],
    )

    trainer.train()

    assert built_states == [(2, 2)]
    assert scheduler.last_epoch == 2
    assert [path.name for path in saved_paths] == [
        "latest_model.tar",
        "model_0002.pth",
    ]


def test_train_with_max_steps_zero_runs_no_epoch_and_saves_nothing(
    tmp_path: Path,
) -> None:
    trainer, scheduler, built_states, saved_paths = _checkpoint_loop_trainer(
        tmp_path,
        epochs=2,
        save_interval=1,
        validation_epoch_interval=0,
        val_losses=[],
    )
    trainer.max_steps = 0
    epoch_calls: list[int] = []
    trainer._train_epoch = lambda epoch, step_callback=None: epoch_calls.append(epoch)

    trainer.train()

    assert epoch_calls == []
    assert trainer.n_steps == 0
    assert scheduler.last_epoch == 0
    assert built_states == []
    assert saved_paths == []


def test_early_stop_checkpoint_uses_post_scheduler_state_once(
    tmp_path: Path,
) -> None:
    """early-stop 保持触发轮次与单次保存，同时固化 scheduler step 后状态。"""
    trainer, scheduler, built_states, saved_paths = _checkpoint_loop_trainer(
        tmp_path,
        epochs=3,
        save_interval=0,
        validation_epoch_interval=1,
        val_losses=[0.5, 0.6],
        patience=1,
    )

    trainer.train()

    assert built_states == [(1, 1), (2, 2)]
    assert scheduler.last_epoch == 2
    assert [path.name for path in saved_paths] == [
        "latest_model.tar",
        "model_0001.pth",
        "best_model.tar",
        "latest_model.tar",
        "model_0002.pth",
    ]


def test_middle_checkpoint_rank0_failure_does_not_reach_barrier(
    tmp_path: Path, fake_distributed
) -> None:
    trainer = _bare_trainer(tmp_path, rank=0, world_size=2)
    trainer._build_state = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("state build failed")
    )

    with pytest.raises(RuntimeError, match="state build failed"):
        trainer._save_middle_checkpoint(epoch=2, batch=5)

    assert fake_distributed == {
        "all_reduce": 1,
        "all_gather_object": 2,
        "barrier": 0,
    }


def test_step_best_rank0_failure_does_not_reach_barrier(
    tmp_path: Path, fake_distributed
) -> None:
    trainer = _bare_trainer(tmp_path, rank=0, world_size=2)
    trainer.val_loader = [object()]
    trainer.validation_steps_interval = 1
    trainer.save_top_k = 1
    trainer.models = {}
    trainer._is_multi_scheduler = False
    trainer.scheduler = None
    trainer.lr_per_step = True
    trainer.best_score = float("inf")
    trainer.best_epoch = -1
    trainer.best_at_step = None
    trainer._curve_x_is_step = False
    trainer._step_best_list = []
    trainer._validation_epoch = lambda _epoch: 0.5
    trainer._log_scalar = lambda *_args, **_kwargs: None
    trainer._build_state = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        OSError("step-best disk full")
    )

    with pytest.raises(RuntimeError, match="step-best disk full"):
        trainer._maybe_validate_and_save_step_best(epoch=2, step=1)

    assert fake_distributed == {
        "all_reduce": 2,
        "all_gather_object": 2,
        "barrier": 0,
    }


def test_load_pretrained_gathers_local_complete_outcomes_without_barriers(
    tmp_path: Path, fake_distributed, monkeypatch
) -> None:
    trainer = _bare_trainer(tmp_path, rank=0, world_size=2)
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"checkpoint")
    loaded: list[tuple[Path, str | None]] = []
    trainer._load_pretrained_local = lambda path, key: (
        loaded.append((path, key)) or (key, 2, ("a", "b"))
    )

    def gather_success(output, value):
        fake_distributed["all_gather_object"] += 1
        output[:] = [value, value]

    monkeypatch.setattr(base_trainer_module.dist, "all_gather_object", gather_success)
    trainer.load_pretrained(str(checkpoint), model_key="generator")

    assert loaded == [(checkpoint, "generator")]
    assert fake_distributed["all_gather_object"] == 1
    assert fake_distributed["barrier"] == 0


def test_resume_gathers_rank_local_failure_and_raises_consistently(
    tmp_path: Path, fake_distributed, monkeypatch
) -> None:
    trainer = _bare_trainer(tmp_path, rank=0, world_size=2)
    checkpoint = tmp_path / "latest_model.tar"
    checkpoint.write_bytes(b"checkpoint")
    trainer._resume_from = str(checkpoint)
    trainer._load_resume_checkpoint_local = lambda _path: (3, 7, 0, 4, 7)

    def gather_rank_failure(output, value):
        fake_distributed["all_gather_object"] += 1
        output[:] = [value, ("error", "OSError", "rank-local read failed")]

    monkeypatch.setattr(base_trainer_module.dist, "all_gather_object", gather_rank_failure)
    with pytest.raises(RuntimeError, match="rank 1: OSError: rank-local read failed"):
        trainer._resume_checkpoint()

    assert fake_distributed["all_gather_object"] == 1
    assert fake_distributed["barrier"] == 0


def test_checkpoint_load_rejects_success_result_mismatch_on_every_rank(
    tmp_path: Path, fake_distributed, monkeypatch
) -> None:
    trainer = _bare_trainer(tmp_path, rank=0, world_size=2)
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"checkpoint")
    trainer._load_pretrained_local = lambda _path, _key: (None, 2, ("a", "b"))

    def gather_mismatch(output, value):
        fake_distributed["all_gather_object"] += 1
        output[:] = [value, ("ok", value[1], (None, 3, ("a", "b", "c")))]

    monkeypatch.setattr(base_trainer_module.dist, "all_gather_object", gather_mismatch)
    with pytest.raises(RuntimeError, match="results differ across ranks"):
        trainer.load_pretrained(str(checkpoint))

    assert fake_distributed["all_gather_object"] == 1
    assert fake_distributed["barrier"] == 0


def test_checkpoint_alias_transaction_rolls_back_partial_publish(
    tmp_path: Path, monkeypatch
) -> None:
    """任一 alias 发布失败时，已存在 alias 必须全部保持旧 generation。"""
    trainer = _bare_trainer(tmp_path)
    latest = tmp_path / "latest_model.tar"
    epoch_path = tmp_path / "model_0001.pth"
    torch.save({"checkpoint_generation": "old", "value": 1}, latest)
    torch.save({"checkpoint_generation": "old", "value": 1}, epoch_path)
    real_replace = base_trainer_module.os.replace
    publish_count = 0

    def fail_second_publish(source, target):
        nonlocal publish_count
        target_path = Path(target)
        if target_path in {latest, epoch_path}:
            publish_count += 1
            if publish_count == 2:
                raise OSError("simulated alias publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(base_trainer_module.os, "replace", fail_second_publish)

    with pytest.raises(OSError, match="alias publish failure"):
        trainer._atomic_save_aliases(
            {"value": 2}, [latest, epoch_path], generation="new"
        )

    assert trainer._safe_load(latest)["checkpoint_generation"] == "old"
    assert trainer._safe_load(epoch_path)["checkpoint_generation"] == "old"
    assert not list(tmp_path.glob(".*.checkpoint-*"))


def test_checkpoint_alias_backup_failure_preserves_unvisited_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    """备份阶段失败不得删除尚未访问的既有 alias（备份走硬链接，copy2 为回退）。"""
    import shutil

    trainer = _bare_trainer(tmp_path)
    aliases = [
        tmp_path / "latest_model.tar",
        tmp_path / "model_0001.pth",
        tmp_path / "best_model.tar",
    ]
    for path in aliases:
        torch.save({"checkpoint_generation": "old", "path": path.name}, path)
    real_link = base_trainer_module.os.link
    real_copy2 = shutil.copy2
    backup_count = 0

    def fail_second_backup_link(source, destination, *args, **kwargs):
        nonlocal backup_count
        if "-backup.tmp" in str(destination):
            backup_count += 1
            if backup_count == 2:
                raise OSError("simulated alias backup failure")
        return real_link(source, destination, *args, **kwargs)

    def reject_backup_copy(source, destination, *args, **kwargs):
        # 硬链接是备份主路径；回退路径也必须失败才能复现「备份阶段失败」。
        if "-backup.tmp" in str(destination):
            raise OSError("simulated alias backup failure")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(base_trainer_module.os, "link", fail_second_backup_link)
    monkeypatch.setattr(shutil, "copy2", reject_backup_copy)

    with pytest.raises(OSError, match="alias backup failure"):
        trainer._atomic_save_aliases({"value": 2}, aliases, generation="new")

    for path in aliases:
        state = trainer._safe_load(path)
        assert state == {"checkpoint_generation": "old", "path": path.name}
    assert not list(tmp_path.glob(".*.checkpoint-*"))


def test_checkpoint_alias_publish_serializes_state_once(
    tmp_path: Path, monkeypatch
) -> None:
    """多 alias 发布只序列化一次，其余 alias 复用同一份数据（硬链接）。"""
    trainer = _bare_trainer(tmp_path)
    latest = tmp_path / "latest_model.tar"
    epoch_path = tmp_path / "model_0001.pth"
    real_save = torch.save
    serialized: list[Path] = []

    def counting_save(state, target, *args, **kwargs):
        serialized.append(Path(target))
        return real_save(state, target, *args, **kwargs)

    monkeypatch.setattr(base_trainer_module.torch, "save", counting_save)

    trainer._atomic_save_aliases({"value": 7}, [latest, epoch_path])

    assert len(serialized) == 1
    assert trainer._safe_load(latest)["value"] == 7
    assert trainer._safe_load(epoch_path)["value"] == 7
    assert latest.stat().st_ino == epoch_path.stat().st_ino
    assert not list(tmp_path.glob(".*.checkpoint-*"))


def test_experiment_context_provenance_never_overwrites_history(tmp_path: Path) -> None:
    """同一 run 目录重复 resume 时 provenance 必须新增文件，不得覆盖首次记录。"""
    trainer = _resume_trainer(tmp_path)
    trainer._save_experiment_context()
    first_git = tmp_path / "git_info.json"
    first_env = tmp_path / "environment.json"
    git_content = first_git.read_text(encoding="utf-8")
    env_content = first_env.read_text(encoding="utf-8")

    trainer._save_experiment_context()
    assert (tmp_path / "git_info_resumed.json").exists()
    assert (tmp_path / "environment_resumed.json").exists()

    trainer._save_experiment_context()
    assert (tmp_path / "git_info_resumed_1.json").exists()
    assert (tmp_path / "environment_resumed_1.json").exists()
    assert first_git.read_text(encoding="utf-8") == git_content
    assert first_env.read_text(encoding="utf-8") == env_content


def test_checkpoint_alias_rollback_failure_retains_recovery_backup(
    tmp_path: Path, monkeypatch
) -> None:
    """回滚失败时不得清理唯一旧副本，且必须显式报告不完整事务。"""
    trainer = _bare_trainer(tmp_path)
    latest = tmp_path / "latest_model.tar"
    epoch_path = tmp_path / "model_0001.pth"
    for path in (latest, epoch_path):
        torch.save({"checkpoint_generation": "old", "path": path.name}, path)
    real_replace = base_trainer_module.os.replace
    publish_count = 0

    def fail_publish_and_latest_rollback(source, destination):
        nonlocal publish_count
        source_path = Path(source)
        destination_path = Path(destination)
        if "-backup.tmp" in source_path.name and destination_path == latest:
            raise OSError("simulated alias rollback failure")
        if destination_path in {latest, epoch_path} and "-backup.tmp" not in source_path.name:
            publish_count += 1
            if publish_count == 2:
                raise OSError("simulated alias publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        base_trainer_module.os, "replace", fail_publish_and_latest_rollback
    )

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        trainer._atomic_save_aliases(
            {"value": 2}, [latest, epoch_path], generation="new"
        )

    recovery_backups = list(
        tmp_path.glob(".latest_model.tar.checkpoint-new-backup.tmp")
    )
    assert len(recovery_backups) == 1
    assert trainer._safe_load(recovery_backups[0])["checkpoint_generation"] == "old"
    assert trainer._safe_load(epoch_path)["checkpoint_generation"] == "old"


def test_atomic_save_cross_device_fallback_replaces_from_target_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """EXDEV fallback 仍须先落目标目录临时文件，再 replace 最终路径。"""
    target = tmp_path / "checkpoint.tar"
    real_replace = base_trainer_module.os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def replace_with_one_exdev(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        replace_calls.append((source_path, destination_path))
        if len(replace_calls) == 1:
            error = OSError("cross-device")
            error.errno = 18
            raise error
        return real_replace(source, destination)

    monkeypatch.setattr(base_trainer_module.os, "replace", replace_with_one_exdev)
    BaseTrainer._atomic_save({"value": 3}, target)

    assert BaseTrainer._safe_load(target)["value"] == 3
    assert replace_calls[-1][0].parent == target.parent
    assert replace_calls[-1][1] == target


def test_resume_failure_rolls_back_model_optimizer_and_progress(tmp_path: Path) -> None:
    """optimizer state 无效时，先前已加载的模型和进度字段不得部分污染。"""
    trainer = _resume_trainer(tmp_path)
    checkpoint = tmp_path / "invalid_optimizer.tar"
    _write_resume_checkpoint(trainer, checkpoint)
    state = trainer._safe_load(checkpoint)
    state["model"] = {
        key: torch.full_like(value, 9.0) for key, value in state["model"].items()
    }
    state["optimizer"]["param_groups"].append(dict(state["optimizer"]["param_groups"][0]))
    torch.save(state, checkpoint)

    model_before = {
        key: value.clone() for key, value in trainer.model.state_dict().items()
    }
    optimizer_before = trainer.optimizer.state_dict()
    trainer.n_steps = 17
    trainer.start_epoch = 8

    with pytest.raises(ValueError, match="parameter group"):
        trainer._load_resume_checkpoint_local(checkpoint)

    for key, value in model_before.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)
    assert trainer.optimizer.state_dict() == optimizer_before
    assert trainer.n_steps == 17
    assert trainer.start_epoch == 8


def test_resume_hook_failure_rolls_back_complete_live_state(
    tmp_path: Path, monkeypatch
) -> None:
    """恢复末段 hook 失败时 model/optimizer/scheduler/RNG/属性均回滚。"""
    trainer = _resume_trainer(tmp_path)
    checkpoint = tmp_path / "hook_failure.tar"
    _write_resume_checkpoint(trainer, checkpoint)
    checkpoint_state = trainer._safe_load(checkpoint)
    torch.manual_seed(1234)
    checkpoint_state["rng_state"] = {
        "torch": torch.get_rng_state().tolist(),
        "torch_cuda": None,
        "numpy": None,
        "python": None,
    }
    torch.save(checkpoint_state, checkpoint)

    with torch.no_grad():
        trainer.model.linear.weight.add_(3.0)
    trainer.optimizer.param_groups[0]["lr"] = 0.037
    trainer.scheduler.step()
    trainer.current_epoch = 11
    trainer.start_epoch = 9
    trainer.n_steps = 23
    trainer.best_score = 1.25
    trainer.train_losses = [4.0]
    torch.manual_seed(9876)

    model_before = {
        key: value.clone() for key, value in trainer.model.state_dict().items()
    }
    optimizer_before = trainer.optimizer.state_dict()
    scheduler_before = trainer.scheduler.state_dict()
    rng_before = torch.get_rng_state().clone()
    monkeypatch.setattr(
        trainer,
        "_after_resume_checkpoint",
        lambda _state: (_ for _ in ()).throw(RuntimeError("injected resume hook failure")),
    )

    with pytest.raises(RuntimeError, match="injected resume hook failure"):
        trainer._load_resume_checkpoint_local(checkpoint)

    for key, value in model_before.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)
    optimizer_after = trainer.optimizer.state_dict()
    assert optimizer_after["param_groups"] == optimizer_before["param_groups"]
    assert optimizer_after["state"].keys() == optimizer_before["state"].keys()
    for parameter_id, before_state in optimizer_before["state"].items():
        for key, before_value in before_state.items():
            after_value = optimizer_after["state"][parameter_id][key]
            if torch.is_tensor(before_value):
                torch.testing.assert_close(after_value, before_value)
            else:
                assert after_value == before_value
    assert trainer.scheduler.state_dict() == scheduler_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert trainer.current_epoch == 11
    assert trainer.start_epoch == 9
    assert trainer.n_steps == 23
    assert trainer.best_score == pytest.approx(1.25)
    assert trainer.train_losses == [4.0]
    assert "_resume_skip_batches" not in trainer.__dict__


def test_metrics_checkpoint_state_round_trips_epoch_count_and_histories(
    monkeypatch, tmp_path: Path
) -> None:
    from core.trainer.metrics_trainer import MetricsCapableTrainer

    trainer = object.__new__(MetricsCapableTrainer)
    trainer._vm_epoch_count = 7
    trainer.val_metrics_epochs = [2, 4]
    trainer.val_metrics_history = {
        "SI-SDR": [1.0, 2.0],
        "DNSMOS_OVRL": [3.25, 3.5],
    }
    trainer.val_metrics_coverage_history = {
        "SI-SDR": [
            {"eligible_count": 4.0, "valid_count": 4.0, "coverage_ratio": 1.0},
            {"eligible_count": 4.0, "valid_count": 3.0, "coverage_ratio": 0.75},
        ]
    }
    trainer.val_metrics_loss_comp_hist = {"sisdr": [0.5, 0.25]}
    trainer._train_loss_comp_hist = {"sisdr": [1.5, 1.0]}
    trainer._val_loss_comp_hist = {"sisdr": [1.25, 0.75]}
    trainer._curve_comp_names = ["sisdr"]
    monkeypatch.setattr(BaseTrainer, "_build_state", lambda self, epoch, **kw: {})

    state = trainer._build_state(4)
    restored = object.__new__(MetricsCapableTrainer)
    restored._after_resume_checkpoint(state)

    assert restored._vm_epoch_count == 7
    assert restored.val_metrics_epochs == [2, 4]
    assert restored.val_metrics_history == {
        "SI-SDR": [1.0, 2.0],
        "DNSMOS_OVRL": [3.25, 3.5],
    }
    assert restored.val_metrics_coverage_history == {
        "SI-SDR": [
            {"eligible_count": 4.0, "valid_count": 4.0, "coverage_ratio": 1.0},
            {"eligible_count": 4.0, "valid_count": 3.0, "coverage_ratio": 0.75},
        ]
    }
    assert restored.val_metrics_loss_comp_hist == {"sisdr": [0.5, 0.25]}
    assert restored._train_loss_comp_hist == {"sisdr": [1.5, 1.0]}
    assert restored._val_loss_comp_hist == {"sisdr": [1.25, 0.75]}


def test_step_best_publishes_checkpoint_and_best_alias_in_one_transaction(
    tmp_path: Path,
) -> None:
    trainer = _bare_trainer(tmp_path)
    trainer.save_top_k = 1
    trainer._step_best_list = []
    trainer._build_state = lambda epoch, **_kwargs: {"epoch": epoch}
    published: list[list[Path]] = []
    trainer._atomic_save_aliases = (
        lambda _state, paths, **_kwargs: published.append(list(paths))
    )

    trainer._save_step_best(epoch=2, step=7, val_loss=0.5, is_new_best=True)

    assert [[path.name for path in paths] for paths in published] == [[
        "best_step=000007-val_loss=0.500.tar",
        "best_model.tar",
    ]]


def test_find_latest_middle_checkpoint_across_runs_uses_epoch_batch_key(
    tmp_path: Path,
) -> None:
    lexically_later = tmp_path / "z-run" / "model_E0002_B999999.pth"
    semantically_later = tmp_path / "a-run" / "model_E0010_B000001.pth"
    lexically_later.parent.mkdir()
    semantically_later.parent.mkdir()
    lexically_later.touch()
    semantically_later.touch()

    assert BaseTrainer._find_latest_middle_checkpoint_across_runs(tmp_path) == (
        semantically_later
    )


def test_resume_lr_override_rebases_scheduler_without_resetting_progress(
    tmp_path: Path,
) -> None:
    trainer = _resume_trainer(tmp_path)
    checkpoint = tmp_path / "lr_override.tar"
    _write_resume_checkpoint(trainer, checkpoint)
    state = trainer._safe_load(checkpoint)
    state["config"] = {"optimizer": {"args": {"lr": 0.1}}}
    torch.save(state, checkpoint)
    trainer.config = {"optimizer": {"args": {"lr": 0.2}}}

    trainer._load_resume_checkpoint_local(checkpoint)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    assert trainer.scheduler.base_lrs == pytest.approx([0.2])
    assert trainer.scheduler.get_last_lr() == pytest.approx([0.2])
    assert trainer.scheduler.last_epoch == 1
    trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.02)


def test_resume_lr_override_rebases_absolute_lr_scheduler_boundaries(
    tmp_path: Path,
) -> None:
    """自定义绝对 LR 调度器恢复后不得在下一 step 跳回旧量级。"""
    from core.optim.scheduler import LinearWarmupCosineAnnealing

    trainer = _resume_trainer(tmp_path)
    trainer.scheduler = LinearWarmupCosineAnnealing(
        trainer.optimizer,
        warmup_steps=4,
        decay_until_step=20,
        max_lr=0.1,
        min_lr=0.01,
    )
    trainer.schedulers = {"default": trainer.scheduler}
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.scheduler.step()
    checkpoint = tmp_path / "absolute_lr_override.tar"
    torch.save(
        {
            "epoch": 2,
            "n_steps": 4,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "config": {"optimizer": {"args": {"lr": 0.1}}},
        },
        checkpoint,
    )
    trainer.config = {"optimizer": {"args": {"lr": 0.2}}}

    trainer._load_resume_checkpoint_local(checkpoint)

    assert trainer.scheduler.last_epoch == 1
    assert trainer.scheduler.base_lrs == pytest.approx([0.2])
    assert trainer.scheduler.max_lr == pytest.approx(0.2)
    assert trainer.scheduler.min_lr == pytest.approx(0.02)
    trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.15)


def test_per_rank_rng_resume_selects_current_rank_and_rejects_world_size_drift(
    tmp_path: Path,
) -> None:
    trainer = _bare_trainer(tmp_path, rank=1, world_size=2)
    states = [{"python": "rank0"}, {"python": "rank1"}]
    checkpoint = {
        "rng_state": states[0],
        "rng_states_by_rank": states,
        "rng_world_size": 2,
    }

    assert trainer._rng_state_for_current_rank(checkpoint) == states[1]

    trainer.world_size = 1
    with pytest.raises(ValueError, match="world size mismatch"):
        trainer._rng_state_for_current_rank(checkpoint)


def test_invalid_epoch_validation_does_not_commit_training_state(tmp_path: Path) -> None:
    trainer, _scheduler, _built_states, _saved_paths = _checkpoint_loop_trainer(
        tmp_path,
        epochs=1,
        save_interval=0,
        validation_epoch_interval=1,
        val_losses=[float("nan")],
        patience=1,
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
    trainer.scheduler = plateau

    trainer.train()

    assert trainer.train_losses == []
    assert trainer.val_losses == []
    assert trainer.curve_epochs == []
    assert trainer.best_score == float("inf")
    assert trainer.best_epoch == -1
    assert trainer._patience_counter == 0
    assert plateau.last_epoch == 0


def test_step_best_checkpoint_records_incomplete_batch_cursor(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)
    trainer.save_top_k = 1
    trainer._step_best_list = []
    captured = {}
    trainer._build_state = lambda epoch, **kwargs: captured.update(kwargs) or {"epoch": epoch}
    trainer._atomic_save_aliases = lambda *_args, **_kwargs: None

    trainer._save_step_best(epoch=2, step=7, val_loss=0.5, batch_cursor=6)

    assert captured == {
        "batch_cursor": 6,
        "epoch_incomplete": True,
        "rng_states_by_rank": None,
    }


def test_resume_optimizer_weight_decay_preserves_no_decay_group(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)
    trainer.optimizer = torch.optim.AdamW(
        [
            {"params": [trainer.model.linear.weight], "weight_decay": 0.1},
            {"params": [trainer.model.linear.bias], "weight_decay": 0.0},
        ],
        lr=0.1,
    )
    trainer.rank = 0
    trainer.config = {"optimizer": {"args": {"lr": 0.1, "weight_decay": 0.2}}}
    trainer.optimizer.param_groups[0]["weight_decay"] = 0.1
    trainer.optimizer.param_groups[1]["weight_decay"] = 0.0

    trainer._override_optimizer_hp_from_config(
        trainer.optimizer,
        None,
        {"optimizer": {"args": {"lr": 0.1, "weight_decay": 0.1}}},
        configured_weight_decays=[0.2, 0.0],
    )

    assert [group["weight_decay"] for group in trainer.optimizer.param_groups] == [0.2, 0.0]


def test_resume_rejects_optimizer_state_shape_and_dtype_mismatch(tmp_path: Path) -> None:
    trainer = _resume_trainer(tmp_path)
    parameter = trainer.model.linear.weight
    parameter.grad = torch.ones_like(parameter)
    trainer.optimizer.step()
    state = trainer.optimizer.state_dict()
    state["state"][0]["momentum_buffer"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        trainer._validate_optimizer_state_dict(
            state, trainer.optimizer, context="test optimizer"
        )

    state = trainer.optimizer.state_dict()
    state["state"][0]["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float64)
    with pytest.raises(TypeError, match="dtype mismatch"):
        trainer._validate_optimizer_state_dict(
            state, trainer.optimizer, context="test optimizer"
        )


def test_resume_rejects_ema_shadow_contract_mismatch(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)
    shadows = [torch.zeros_like(parameter) for parameter in trainer.model.parameters()]
    trainer._validate_ema_state({"shadow_params": shadows})
    with pytest.raises(ValueError, match="count mismatch"):
        trainer._validate_ema_state({"shadow_params": []})
    wrong_dtype = list(shadows)
    wrong_dtype[0] = wrong_dtype[0].to(torch.float64)
    with pytest.raises(TypeError, match="dtype mismatch"):
        trainer._validate_ema_state({"shadow_params": wrong_dtype})


def test_resume_outer_outcome_failure_rolls_back_live_state(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = _resume_trainer(tmp_path)
    checkpoint = tmp_path / "latest_model.tar"
    checkpoint.write_bytes(b"checkpoint")
    trainer._resume_from = str(checkpoint)
    trainer.world_size = 2
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    monkeypatch.setattr(base_trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(base_trainer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        base_trainer_module.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value, ("ok", "mismatch")]),
    )
    monkeypatch.setattr(
        trainer,
        "_checkpoint_content_identity",
        lambda _path: (1, "a" * 64),
    )

    checkpoint_state = {
        "epoch": 2,
        "n_steps": 4,
        "model": {key: torch.full_like(value, 9.0) for key, value in before.items()},
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
    }
    monkeypatch.setattr(trainer, "_safe_load", lambda _path: checkpoint_state)
    with pytest.raises(RuntimeError, match="results differ across ranks"):
        trainer._resume_checkpoint()
    for key, value in before.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)


def test_validation_metric_component_history_uses_current_validation_values() -> None:
    from core.trainer.metrics_trainer import MetricsCapableTrainer

    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer._vm_metrics = ["SI-SDR"]
    trainer._vm_enabled = True
    trainer._vm_strict = False
    trainer._vm_min_coverage = {}
    trainer._comp_keys = ["current"]
    trainer._last_val_comp = {"current": 1.0}
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_coverage_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    trainer._should_save_curve = lambda _count: False
    trainer._log_scalar = lambda *_args, **_kwargs: None

    trainer._log_val_metrics(
        None,
        1,
        aggregated_intrusive={"SI-SDR": 2.0, "_count_SI-SDR": 1.0},
        eligible_count=1,
        val_loss_components={"current": 3.0},
    )

    assert trainer.val_metrics_loss_comp_hist == {"current": [3.0]}


def test_aborted_validation_metrics_commit_no_logging_or_history() -> None:
    from core.trainer.metrics_trainer import MetricsCapableTrainer

    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer._vm_metrics = ["SI-SDR"]
    trainer._vm_enabled = True
    trainer._vm_strict = False
    trainer._vm_min_coverage = {}
    trainer._comp_keys = ["current"]
    trainer._last_val_comp = {"current": 1.0}
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_coverage_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    logged = []
    trainer._log_scalar = lambda *args, **kwargs: logged.append((args, kwargs))
    trainer._should_save_curve = lambda _count: pytest.fail("aborted metrics saved a curve")

    trainer._log_val_metrics(
        None,
        1,
        aggregated_intrusive={"SI-SDR": 2.0, "_count_SI-SDR": 1.0},
        eligible_count=1,
        metrics_aborted=True,
        val_loss_components={"current": 3.0},
    )

    assert logged == []
    assert trainer.val_metrics_epochs == []
    assert trainer.val_metrics_history == {}
    assert trainer.val_metrics_coverage_history == {}
    assert trainer.val_metrics_loss_comp_hist == {}


pytestmark = [pytest.mark.unit]
