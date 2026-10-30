"""共享训练正确性回归测试。"""

from __future__ import annotations

import contextlib
import math
from datetime import timedelta
from pathlib import Path

import numpy as np

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.data.post_augment import apply_post_augmentations
from core.data.simulation_configs import SimulationConfigs
from core.losses.state_ce import StateCrossEntropyLoss
from core.trainer.metrics_trainer import (
    MetricsCapableTrainer,
    validate_waveform_lengths,
)
from tasks.se.data.batch import SEBatch
from tasks.se.trainer.dual_input_loss_trainer import DualInputLossTrainer
from tasks.se.trainer.gan_trainer import GANTrainer
from tasks.tse.cue_contract import TSECueBatch, TSECueBatches, TSECueSpec, TSECueSpecs
from tasks.tse.data.batch import TSEBatch
from tasks.tse.trainer.tse_trainer import TSETrainer


class _BreakdownLoss:
    _last_breakdown = [("part", 1.0, 1.0, 1.0)]


def _validation_harness(batches, *, max_val_batches: int = 0):
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.world_size = 1
    trainer.val_loader = batches
    trainer.max_val_batches = max_val_batches
    trainer._comp_keys = ["part"]
    trainer.loss_fn = _BreakdownLoss()
    trainer._vm_enabled = True
    trainer._vm_metrics = ["SI-SDR"]
    trainer._vm_interval = 1
    trainer._vm_epoch_count = 0
    trainer._vm_max_metrics = 100
    trainer._curve_x_is_step = False
    trainer.n_steps = 0
    trainer._last_val_comp = {}
    trainer._ema_context = contextlib.nullcontext
    trainer._val_iter_is_ddp_padding = lambda _n: False
    trainer._init_val_extra_state = lambda: None
    trainer._val_accumulate_extra = lambda *_args: None
    trainer._val_post_allreduce_extra = lambda *_args: None
    trainer._val_log_extra_metrics = lambda *_args: None
    trainer._val_run_extra_metrics = lambda *_args: None
    trainer._log_output_sanitize_summary = lambda *_args: None
    trainer.logged_scalars = {}
    trainer._log_scalar = lambda name, value, step: trainer.logged_scalars.__setitem__(
        name, (value, step)
    )
    trainer._log_val_loss_summary = lambda *args: setattr(
        trainer, "logged_summary", args
    )

    def record_metric_pairs(*args, **kwargs):
        trainer.metric_pairs = list(args[1])
        trainer.metric_kwargs = kwargs
        return False

    trainer._run_validation_metrics_from_pairs = record_metric_pairs

    def forward(raw_batch, _epoch, _n_loops):
        loss_value, rows = raw_batch
        trainer.loss_fn._last_breakdown = [
            ("part", 1.0, loss_value, loss_value)
        ]
        clean = torch.ones(rows, 8)
        return (
            torch.tensor(float(loss_value)),
            clean.clone(),
            clean,
            torch.full((rows,), 8, dtype=torch.long),
            16000,
            None,
        )

    trainer._val_forward_loss = forward
    return trainer


@pytest.mark.unit
def test_validation_total_and_composite_are_weighted_by_input_rows():
    trainer = _validation_harness([(1.0, 2), (3.0, 1)])

    avg = trainer._validation_epoch(epoch=4)

    assert avg == pytest.approx(5.0 / 3.0)
    _, logged_avg, n_global, comp_sums, *_ = trainer.logged_summary
    assert logged_avg == pytest.approx(5.0 / 3.0)
    assert n_global == 3
    assert comp_sums["part"] == pytest.approx(5.0)


@pytest.mark.unit
def test_max_val_batches_still_collects_metrics_for_last_allowed_batch():
    trainer = _validation_harness([(1.0, 2), (3.0, 1)], max_val_batches=1)

    trainer._validation_epoch(epoch=0)

    assert len(trainer.metric_pairs) == 2


@pytest.mark.unit
def test_partial_nonfinite_val_loss_keeps_all_objective_metric_pairs():
    trainer = _validation_harness([(float("nan"), 2), (3.0, 1)])

    avg = trainer._validation_epoch(epoch=5)

    assert avg == pytest.approx(3.0)
    assert len(trainer.metric_pairs) == 3
    assert trainer.metric_kwargs["commit_history"] is True
    assert trainer.logged_scalars["val/finite_loss_samples"] == (1.0, 5)
    assert trainer.logged_scalars["val/total_samples"] == (3.0, 5)
    assert trainer.logged_scalars["val/finite_loss_coverage_ratio"] == pytest.approx(
        (1.0 / 3.0, 5)
    )


@pytest.mark.unit
def test_all_nonfinite_val_loss_suppresses_objective_metric_history_commit():
    trainer = _validation_harness([(float("nan"), 2), (float("inf"), 1)])

    avg = trainer._validation_epoch(epoch=6)

    assert math.isnan(avg)
    assert len(trainer.metric_pairs) == 3
    assert trainer.metric_kwargs["commit_history"] is False
    assert trainer.logged_scalars["val/finite_loss_samples"] == (0.0, 6)
    assert trainer.logged_scalars["val/total_samples"] == (3.0, 6)


@pytest.mark.unit
def test_val_metric_coverage_logs_eligible_valid_and_ratio():
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer._vm_metrics = ["SI-SDR", "PESQ"]
    trainer._vm_parallel_intrusive = False
    trainer._vm_metrics_workers = 0
    trainer._vm_strict = False
    trainer._vm_enabled = True
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    trainer._comp_keys = []
    trainer._nccl_flow_log = lambda *_args: None
    trainer._allreduce_sum = lambda value, **_kwargs: value
    scalars = {}
    trainer._log_scalar = lambda name, value, step: scalars.__setitem__(
        name, (value, step)
    )
    trainer._should_save_curve = lambda _count: False

    trainer._log_val_metrics(
        None,
        epoch=7,
        aggregated_intrusive={
            "SI-SDR": 6.0,
            "_count_SI-SDR": 3.0,
            "PESQ": 2.5,
            "_count_PESQ": 1.0,
        },
        eligible_count=4,
    )

    assert scalars["val/SI-SDR/eligible_count"] == (4.0, 7)
    assert scalars["val/SI-SDR/valid_count"] == (3.0, 7)
    assert scalars["val/SI-SDR/coverage_ratio"] == (0.75, 7)
    assert scalars["val/PESQ/eligible_count"] == (4.0, 7)
    assert scalars["val/PESQ/valid_count"] == (1.0, 7)
    assert scalars["val/PESQ/coverage_ratio"] == (0.25, 7)
    assert trainer.val_metrics_epochs == [7]
    assert trainer.val_metrics_history == {"SI-SDR": [2.0], "PESQ": [2.5]}


@pytest.mark.unit
def test_non_strict_val_metrics_keep_nan_hole_without_logging_nan_scalar(tmp_path):
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer._vm_metrics = ["SI-SDR", "PESQ"]
    trainer._vm_parallel_intrusive = False
    trainer._vm_metrics_workers = 0
    trainer._vm_strict = False
    trainer._vm_enabled = True
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_coverage_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    trainer.log_dir = tmp_path
    trainer._comp_keys = []
    trainer._nccl_flow_log = lambda *_args: None
    trainer._allreduce_sum = lambda value, **_kwargs: value
    scalars = {}
    trainer._log_scalar = lambda name, value, step: scalars.__setitem__(
        name, (value, step)
    )
    trainer._should_save_curve = lambda _count: False

    trainer._log_val_metrics(
        None,
        epoch=9,
        aggregated_intrusive={
            "SI-SDR": 6.0,
            "_count_SI-SDR": 3.0,
            "PESQ": 0.0,
            "_count_PESQ": 0.0,
        },
        eligible_count=4,
    )

    assert trainer.val_metrics_epochs == [9]
    assert trainer.val_metrics_history["SI-SDR"] == [2.0]
    assert len(trainer.val_metrics_history["PESQ"]) == 1
    assert math.isnan(trainer.val_metrics_history["PESQ"][0])
    assert "val/PESQ" not in scalars
    assert scalars["val/PESQ/valid_count"] == (0.0, 9)
    assert scalars["val/PESQ/coverage_ratio"] == (0.0, 9)
    assert scalars["val/SI-SDR"] == (2.0, 9)

    trainer._write_val_metrics_csv()
    header, row = (tmp_path / "val_metrics.csv").read_text().splitlines()
    assert header.split(",")[:3] == ["epoch", "PESQ", "SI-SDR"]
    assert row.split(",")[:3] == ["9", "", "2.000000"]


@pytest.mark.unit
def test_val_metric_min_coverage_fails_before_logging_or_history_commit():
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer._vm_metrics = ["SI-SDR", "PESQ"]
    trainer._vm_parallel_intrusive = False
    trainer._vm_metrics_workers = 0
    trainer._vm_strict = False
    trainer._vm_min_coverage = {"PESQ": 0.5}
    trainer._vm_enabled = True
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_coverage_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    trainer._comp_keys = []
    trainer._nccl_flow_log = lambda *_args: None
    trainer._allreduce_sum = lambda value, **_kwargs: value
    logged = []
    trainer._log_scalar = lambda *args: logged.append(args)

    with pytest.raises(RuntimeError, match="PESQ validation coverage.*below"):
        trainer._log_val_metrics(
            None,
            epoch=7,
            aggregated_intrusive={
                "SI-SDR": 6.0,
                "_count_SI-SDR": 3.0,
                "PESQ": 2.5,
                "_count_PESQ": 1.0,
            },
            eligible_count=4,
        )

    assert logged == []
    assert trainer.val_metrics_epochs == []
    assert trainer.val_metrics_history == {}
    assert trainer.val_metrics_coverage_history == {}


@pytest.mark.unit
def test_val_metrics_do_not_commit_history_without_finite_loss_samples():
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer._vm_metrics = ["SI-SDR"]
    trainer._vm_parallel_intrusive = False
    trainer._vm_metrics_workers = 0
    trainer._vm_strict = False
    trainer._vm_enabled = True
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_coverage_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    trainer._comp_keys = []
    trainer._nccl_flow_log = lambda *_args: None
    trainer._allreduce_sum = lambda value, **_kwargs: value
    scalars = {}
    trainer._log_scalar = lambda name, value, step: scalars.__setitem__(
        name, (value, step)
    )

    trainer._log_val_metrics(
        None,
        epoch=8,
        aggregated_intrusive={"SI-SDR": 6.0, "_count_SI-SDR": 3.0},
        eligible_count=4,
        commit_history=False,
    )

    assert trainer.val_metrics_epochs == []
    assert trainer.val_metrics_history == {}
    assert trainer.val_metrics_coverage_history == {}
    assert "val/SI-SDR" not in scalars
    assert scalars["val/SI-SDR/coverage_ratio"] == (0.75, 8)


@pytest.mark.unit
def test_ddp_quota_is_exact_when_global_limit_is_not_divisible():
    quotas = []
    for rank in range(3):
        trainer = object.__new__(MetricsCapableTrainer)
        trainer.rank = rank
        trainer.world_size = 3
        quotas.append(trainer._ddp_rank_quota(5))
    assert quotas == [2, 2, 1]
    assert sum(quotas) == 5

    small = []
    for rank in range(4):
        trainer = object.__new__(MetricsCapableTrainer)
        trainer.rank = rank
        trainer.world_size = 4
        small.append(trainer._ddp_rank_quota(2))
    assert small == [1, 1, 0, 0]
    assert sum(small) == 2


@pytest.mark.unit
def test_ddp_timing_uses_critical_path_and_global_samples(monkeypatch):
    """DDP timing uses slowest-rank wall time, not mean rank time."""
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.device = torch.device("cpu")
    trainer.world_size = 2

    def fake_all_reduce(tensor, op=None):
        if op is torch.distributed.ReduceOp.MAX:
            tensor.copy_(torch.tensor([3.0, 5.0, 11.0], dtype=tensor.dtype))
        else:
            tensor.copy_(torch.tensor([20.0], dtype=tensor.dtype))

    monkeypatch.setattr("core.trainer.metrics_trainer.dist.all_reduce", fake_all_reduce)

    data_wait, compute, samples, elapsed = trainer._allreduce_timing_stats(
        1.0, 2.0, 8, epoch_wall_sec=4.0
    )

    assert data_wait == pytest.approx(3.0)
    assert compute == pytest.approx(5.0)
    assert samples == pytest.approx(20.0)
    assert elapsed == pytest.approx(11.0)
    assert samples / elapsed == pytest.approx(20.0 / 11.0)


@pytest.mark.unit
def test_cpu_timing_uses_epoch_wall_clock_for_throughput():
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.device = torch.device("cpu")
    trainer.world_size = 1

    data_wait, compute, samples, elapsed = trainer._allreduce_timing_stats(
        2.0, 3.0, 4, epoch_wall_sec=10.0
    )

    assert (data_wait, compute, samples, elapsed) == pytest.approx((2.0, 3.0, 4.0, 10.0))
    assert samples / elapsed == pytest.approx(0.4)


@pytest.mark.unit
def test_val_loss_stats_use_one_atomic_allreduce(monkeypatch):
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.device = torch.device("cpu")
    trainer.world_size = 2
    trainer._nccl_flow_log = lambda *_args: None
    calls = []

    def fake_all_reduce(tensor, op=None):
        calls.append((tensor.clone(), op))
        tensor.copy_(torch.tensor([9.0, 4.0, 6.0], dtype=tensor.dtype))

    monkeypatch.setattr("core.trainer.metrics_trainer.dist.all_reduce", fake_all_reduce)

    stats = trainer._allreduce_val_loss_stats(torch.tensor([2.5]), 2, 3)

    assert stats == (9.0, 4.0, 6.0)
    assert len(calls) == 1
    assert calls[0][0].tolist() == [2.5, 2.0, 3.0]
    assert calls[0][1] is torch.distributed.ReduceOp.SUM


@pytest.mark.unit
def test_output_sanitize_counts_are_summed_in_fixed_order(monkeypatch):
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.device = torch.device("cpu")
    trainer.world_size = 2
    trainer._sanitize_n_nonfinite = torch.tensor(2, dtype=torch.int64)
    trainer._sanitize_n_clamped = torch.tensor(3, dtype=torch.int64)
    calls = []

    def fake_all_reduce(tensor, op=None):
        calls.append((tensor.clone(), op))
        tensor.copy_(torch.tensor([11, 13], dtype=tensor.dtype))

    monkeypatch.setattr("core.trainer.metrics_trainer.dist.all_reduce", fake_all_reduce)

    trainer._allreduce_output_sanitize_stats()

    assert len(calls) == 1
    assert calls[0][0].tolist() == [2, 3]
    assert calls[0][1] is torch.distributed.ReduceOp.SUM
    assert trainer._sanitize_n_nonfinite.item() == 11
    assert trainer._sanitize_n_clamped.item() == 13
    assert trainer._sanitize_stats_global is True


@pytest.mark.unit
def test_tse_validation_weight_uses_expanded_target_rows():
    trainer = object.__new__(TSETrainer)
    target = torch.zeros(5, 16)
    assert trainer._val_loss_weight(target, target, torch.ones(5), None) == 5


@pytest.mark.unit
def test_tse_train_sanitize_counts_reach_accumulator_without_host_conversion(monkeypatch):
    """sanitize 计数留在设备侧传递，step 路径不得做 bool/item 同步。"""

    class DeviceCountSentinel:
        def __bool__(self):
            raise AssertionError("sanitize count must not be converted to bool")

        def item(self):
            raise AssertionError("sanitize count must not be copied to host")

    nonfinite_count = DeviceCountSentinel()
    clamped_count = DeviceCountSentinel()
    estimate = torch.full((1, 8), 0.25)
    monkeypatch.setattr(
        "tasks.tse.trainer.tse_trainer.sanitize_model_output",
        lambda output: (output, nonfinite_count, clamped_count),
    )

    trainer = object.__new__(TSETrainer)
    trainer._tse_joint_training = True
    trainer._tse_spkemb_enabled = False
    trainer._dual_self_estimated_loss_weights = None
    trainer._maybe_ssa_cues = lambda batch, epoch, batch_idx: batch.cues
    trainer._prepare_speaker_conditioning = (
        lambda batch, cues, phase: (cues, None)
    )
    trainer._forward_tse_batch = lambda batch, **kwargs: {
        "target_estimate": estimate,
        "mixture": batch.mixture[:, 0],
        "target": batch.target,
        "lengths": batch.lengths,
    }
    accumulated = []
    trainer._accumulate_output_sanitize = lambda *counts: accumulated.append(counts)
    trainer._composite_spk_ce_kwargs = lambda data, labels: {}
    trainer.loss_fn = lambda enhanced, target, lengths, **kwargs: enhanced.mean()
    speaker_waveform = TSECueBatch(
        representation="waveform",
        value=torch.ones(1, 8),
        present=torch.ones(1, dtype=torch.bool),
        lengths=torch.tensor([8], dtype=torch.int32),
        timebase=None,
    )
    batch = TSEBatch(
        mixture=torch.ones(1, 1, 8),
        target=torch.ones(1, 8),
        sr=torch.tensor(16000, dtype=torch.int32),
        lengths=torch.tensor([8], dtype=torch.int32),
        num_targets=torch.ones(1, dtype=torch.int32),
        spk_labels=torch.full((1,), -1, dtype=torch.int64),
        target_present=torch.ones(1, dtype=torch.bool),
        cues=TSECueBatches(speaker_waveform=speaker_waveform),
        cue_specs=TSECueSpecs(
            speaker_waveform=TSECueSpec("waveform", (8,))
        ),
        keys=("sample",),
        speaker_ids=("speaker",),
        geometry=(None,),
        timebase=(None,),
        provenance=(None,),
    )

    loss, loss_was_finite, _ = trainer._forward_and_compute_loss(batch, 0, 0, 16000)

    assert loss_was_finite is True
    assert loss.item() == pytest.approx(0.25)
    assert accumulated == [(nonfinite_count, clamped_count)]


class _DualLossModel(torch.nn.Module):
    def forward(self, noisy, clean, lengths):
        return clean[0, 0]


def _configure_specialized_val_trainer(trainer, batches):
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.world_size = 1
    trainer.val_loader = batches
    trainer.max_val_batches = 0
    trainer.use_amp = False
    trainer.amp_dtype = torch.float32
    trainer._curve_x_is_step = False
    trainer.n_steps = 0
    trainer._ema_context = contextlib.nullcontext
    trainer._val_iter_is_ddp_padding = lambda _n: False
    trainer._log_scalar = lambda *_args: None
    return trainer


@pytest.mark.unit
def test_dual_input_validation_is_weighted_by_sample_rows():
    batches = [
        SEBatch(
            clean=torch.ones(2, 8), noisy=torch.zeros(2, 8),
            sr=torch.tensor(16000), lengths=torch.full((2,), 8),
        ),
        SEBatch(
            clean=torch.full((1, 8), 3.0), noisy=torch.zeros(1, 8),
            sr=torch.tensor(16000), lengths=torch.full((1,), 8),
        ),
    ]
    trainer = _configure_specialized_val_trainer(
        object.__new__(DualInputLossTrainer), batches
    )
    trainer.model = _DualLossModel()
    trainer._model_accepts_fs = False

    assert trainer._validation_epoch(epoch=0) == pytest.approx(5.0 / 3.0)


@pytest.mark.unit
def test_gan_validation_is_weighted_by_sample_rows():
    trainer = _configure_specialized_val_trainer(
        object.__new__(GANTrainer), [(1.0, 2), (3.0, 1)]
    )
    trainer.generator = torch.nn.Identity()
    trainer._vm_max_metrics = 0
    trainer._reset_output_sanitize_stats = lambda: None
    trainer._advance_and_should_run_metrics = lambda: False

    def unpack(batch):
        rows = batch[1]
        clean = torch.ones(rows, 8)
        return {
            "loss": batch[0],
            "rows": rows,
            "clean": clean,
            "lengths": torch.full((rows,), 8),
        }

    trainer._unpack_batch = unpack

    def forward(batch):
        rows = batch["rows"]
        clean = batch["clean"]
        return {
            "loss": batch["loss"], "clean": clean, "enhanced": clean.clone(),
            "lengths": torch.full((rows,), 8), "fs": torch.tensor(16000),
        }

    trainer._forward_generator = forward
    trainer._sanitize_forward_output = lambda fwd: fwd
    trainer._compute_validation_loss = lambda fwd: torch.tensor(float(fwd["loss"]))
    trainer._run_validation_metrics_from_pairs = lambda *_args, **_kwargs: None
    trainer._log_output_sanitize_summary = lambda *_args: None

    assert trainer._validation_epoch(epoch=0) == pytest.approx(5.0 / 3.0)


def _augment_cfg(**overrides):
    cfg = SimulationConfigs()
    cfg.noisy_post_mix_degradations_enabled = True
    cfg.prob_clip = 0.0
    cfg.prob_dc_offset = 1.0
    cfg.dc_offset_range = (0.1, 0.1)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


@pytest.mark.unit
def test_layer_f_cpu_execution_preserves_target_and_facts():
    cfg = _augment_cfg()
    noisy = np.zeros(32, dtype=np.float32)
    target = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    target_before = target.copy()
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
    np.testing.assert_array_equal(target, target_before)
    assert facts == {}


@pytest.mark.unit
def test_layer_f_cpu_execution_respects_padding_and_length_boundary():
    cfg = _augment_cfg()
    valid_length = 16
    noisy = np.zeros(32, dtype=np.float32)
    noisy[valid_length:] = 20.0
    target = np.full(32, -3.0, dtype=np.float32)
    target_before = target.copy()
    output = noisy.copy()
    output[:valid_length] = apply_post_augmentations(
        noisy[:valid_length],
        16_000,
        cfg,
        rng=np.random.default_rng(0),
    )

    assert np.all(output[valid_length:] == noisy[valid_length:])
    np.testing.assert_array_equal(target, target_before)


@pytest.mark.unit
def test_layer_f_cpu_execution_skips_all_ops_when_disabled():
    cfg = _augment_cfg()
    cfg.noisy_post_mix_degradations_enabled = False
    noisy = np.zeros(16, dtype=np.float32)
    operations, facts = [], {"stale": True}

    output = apply_post_augmentations(
        noisy,
        16_000,
        cfg,
        rng=np.random.default_rng(0),
        applied_ops_out=operations,
        clipping_facts_out=facts,
    )

    np.testing.assert_array_equal(output, noisy)
    assert operations == []
    assert facts == {}


@pytest.mark.unit
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_state_ce_all_ignore_returns_connected_exact_zero(nonfinite):
    logits = torch.tensor(
        [[[nonfinite, 1.0, -2.0], [3.0, nonfinite, 4.0]]],
        requires_grad=True,
    )
    state = torch.full((1, 2), -1, dtype=torch.long)

    loss = StateCrossEntropyLoss()(
        torch.zeros(1, 8),
        torch.zeros(1, 8),
        state_probs=logits,
        state=state,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def _gloo_max_steps_boundary_worker(rank: int, init_file: str, result_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = object.__new__(MetricsCapableTrainer)
        trainer.rank = rank
        trainer.world_size = 2
        trainer.device = torch.device("cpu")
        trainer.max_steps = 3
        trainer.n_steps = 3
        trainer._nccl_flow_log = lambda *_args: None
        reached = trainer._sync_max_steps_reached()
        Path(result_dir, f"rank{rank}.txt").write_text(str(reached))
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.ddp
@pytest.mark.subprocess
def test_gloo_max_steps_boundary_stops_all_ranks_together(tmp_path):
    if not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo backend unavailable")
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _gloo_max_steps_boundary_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert (tmp_path / "rank0.txt").read_text() == "True"
    assert (tmp_path / "rank1.txt").read_text() == "True"


class _GlooMaxStepsTrainer(MetricsCapableTrainer):
    """绕过完整初始化、但复用生产 `_train_epoch` 的 CPU DDP harness。"""

    def _unpack_batch(self, raw_batch):
        return raw_batch

    def _batch_size_from_batch(self, batch):
        """测试夹具显式用 input 定义样本轴，不依赖基类字段猜测。"""
        return int(batch["input"].shape[0])

    def _forward_and_compute_loss(self, batch, epoch, batch_idx, fs_val):
        estimate = self.model(batch["input"])
        return torch.nn.functional.mse_loss(estimate, batch["target"]), True, {}

    def _log_scalar(self, *_args, **_kwargs):
        return None


def _gloo_max_steps_train_epoch_worker(
    rank: int, init_file: str, result_dir: str
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(0)
        trainer = object.__new__(_GlooMaxStepsTrainer)
        trainer.rank = rank
        trainer.world_size = 2
        trainer.device = torch.device("cpu")
        trainer.model = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Linear(1, 1)
        )
        trainer.models = {"default": trainer.model}
        trainer._is_multi_model = False
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.optimizers = {"default": trainer.optimizer}
        trainer._is_multi_optimizer = False
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(
            trainer.optimizer, step_size=1
        )
        trainer.schedulers = {"default": trainer.scheduler}
        trainer._is_multi_scheduler = False
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.use_amp = False
        trainer.amp_dtype = torch.float32
        trainer.clip_grad = 0.0
        trainer.ema = None
        trainer.ema_sync_every_steps = 0
        trainer.lr_per_step = False
        trainer._pending_sched_step_keys = []
        trainer.n_steps = 0
        trainer.max_steps = 1
        trainer.accumulation_steps = 2
        trainer.start_epoch = 1
        trainer._resume_skip_batches = 0
        trainer._strict_checks = False
        trainer.log_interval = 100
        trainer.save_batch_interval = 0
        trainer._train_log_loss_win = []
        trainer.loss_fn = torch.nn.MSELoss()
        trainer._comp_keys = None
        trainer.writer = None
        trainer.wandb_run = None
        trainer.sample_rate = 16000
        trainer.train_loader = [
            {
                "input": torch.tensor([[float(index + rank)]]),
                "target": torch.zeros(1, 1),
                "sr": torch.tensor(16000),
            }
            for index in range(3)
        ]

        epoch_loss = trainer._train_epoch(1)
        result = (
            trainer.n_steps,
            trainer._last_completed_batch_cursor,
            trainer._train_epoch_completed,
            epoch_loss,
        )
        Path(result_dir, f"train-rank{rank}.txt").write_text(repr(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.ddp
@pytest.mark.subprocess
def test_gloo_train_epoch_max_steps_stops_at_shared_accumulation_cursor(tmp_path):
    """真实 DDP backward 在 max_steps 边界无死锁，且两 rank cursor 一致。"""
    if not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo backend unavailable")
    init_file = tmp_path / "gloo_train_init"
    mp.spawn(
        _gloo_max_steps_train_epoch_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    expected_prefix = "(1, 2, False, "
    for rank in range(2):
        assert (tmp_path / f"train-rank{rank}.txt").read_text().startswith(
            expected_prefix
        )


@pytest.mark.unit
def test_validate_waveform_lengths_rejects_non_integer_and_out_of_range() -> None:
    validate_waveform_lengths(
        torch.tensor([4, 8], dtype=torch.int64),
        batch_size=2,
        max_length=8,
        context="test",
    )
    with pytest.raises(TypeError, match="integer dtype"):
        validate_waveform_lengths(
            torch.tensor([4.0, 8.0]),
            batch_size=2,
            max_length=8,
            context="test",
        )
    with pytest.raises(ValueError, match=r"\[1, 8\]"):
        validate_waveform_lengths(
            torch.tensor([0, 9]),
            batch_size=2,
            max_length=8,
            context="test",
        )


@pytest.mark.unit
def test_strict_val_metrics_requires_every_configured_metric() -> None:
    trainer = object.__new__(MetricsCapableTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer._vm_metrics = ["SI-SDR", "PESQ"]
    trainer._vm_parallel_intrusive = False
    trainer._vm_metrics_workers = 0
    trainer._vm_strict = True
    trainer._vm_enabled = True
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    trainer._comp_keys = []
    trainer._nccl_flow_log = lambda *_args: None
    trainer._allreduce_sum = lambda value, **_kwargs: value
    trainer._log_scalar = lambda *_args: None

    with pytest.raises(RuntimeError, match="PESQ"):
        trainer._log_val_metrics(
            None,
            epoch=1,
            aggregated_intrusive={"SI-SDR": 2.0, "_count_SI-SDR": 1.0},
        )

    assert trainer.val_metrics_epochs == []
    assert trainer.val_metrics_history == {}
