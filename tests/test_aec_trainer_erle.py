"""AEC 最终复核合同的回归测试。"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tasks.aec.data.linear_aec_frontend import LinearAECFrontend
from tasks.aec.inference.offline_alignment import align_farend_to_mic
from tasks.aec.trainer.aec_trainer import AECTrainer
from core.metrics.aec import compute_erle, compute_erle_energy_sums


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _ImmediateExecutor:
    """同步执行的 ProcessPoolExecutor 替身，保留 submit/shutdown 契约。"""

    def __init__(self, *, max_workers, mp_context):
        self.max_workers = max_workers
        self.mp_context = mp_context
        self._processes = {}
        self.shutdown_calls = []

    def submit(self, fn, task):
        return _ImmediateFuture(fn(task))

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


def _wait_all(pending, **_kwargs):
    return set(pending), set()


@pytest.mark.unit
def test_erle_far_gate_is_shifted_to_mic_timeline():
    fs = 1000
    far = np.zeros(100, dtype=np.float32)
    far[10:20] = 1.0
    near = np.zeros_like(far)
    mic = np.zeros_like(far)
    enh = np.zeros_like(far)
    mic[10:20] = 10.0  # 未补偿 gate 会错误统计这段干扰。
    enh[10:20] = 1.0
    mic[40:50] = 2.0
    enh[40:50] = 1.0

    delayed = compute_erle_energy_sums(
        mic,
        near,
        enh,
        fs=fs,
        far=far,
        gate_frame_ms=1.0,
        gate_hop_ms=1.0,
        far_delay_samples=30,
    )
    unshifted = compute_erle_energy_sums(
        mic,
        near,
        enh,
        fs=fs,
        far=far,
        gate_frame_ms=1.0,
        gate_hop_ms=1.0,
        far_delay_samples=0,
    )

    assert delayed == pytest.approx((40.0, 10.0, 10))
    assert unshifted == pytest.approx((1000.0, 10.0, 10))


@pytest.mark.unit
def test_erle_automatic_gcc_delay_matches_explicit_delay():
    fs = 1000
    rng = np.random.default_rng(7)
    far = np.zeros(600, dtype=np.float32)
    far[100:300] = rng.normal(size=200).astype(np.float32)
    delay = 37
    mic = np.zeros_like(far)
    mic[delay:] = far[:-delay]
    near = np.zeros_like(far)
    enh = mic * 0.25

    automatic = compute_erle_energy_sums(
        mic, near, enh, fs=fs, far=far, gate_frame_ms=1.0, gate_hop_ms=1.0
    )
    explicit = compute_erle_energy_sums(
        mic,
        near,
        enh,
        fs=fs,
        far=far,
        gate_frame_ms=1.0,
        gate_hop_ms=1.0,
        far_delay_samples=delay,
    )

    assert automatic == pytest.approx(explicit)
    assert compute_erle(mic, near, enh, fs=fs, far=far) == pytest.approx(
        compute_erle(mic, near, enh, fs=fs, far=far, far_delay_samples=delay)
    )


def _erle_logging_trainer(*, world_size=1):
    trainer = object.__new__(AECTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.world_size = world_size
    trainer.logged = []
    trainer._log_scalar = lambda name, value, step: trainer.logged.append(
        (name, value, step)
    )
    return trainer


@pytest.mark.unit
def test_trainer_erle_keeps_legacy_macro_and_adds_energy_micro():
    trainer = _erle_logging_trainer()
    stats = {
        "erle_db_sum": 26.0,
        "erle_count": 2,
        "erle_energy_sum_in": 100.0,
        "erle_energy_sum_out": 10.0,
        "erle_energy_count": 20,
    }

    trainer._compute_and_log_erle(epoch=4, erle_stats=stats)

    logged = {name: value for name, value, step in trainer.logged if step == 4}
    assert logged["val/ERLE"] == pytest.approx(13.0)
    assert logged["val/ERLE_residual_macro"] == pytest.approx(13.0)
    assert logged["val/ERLE_energy_micro"] == pytest.approx(10.0)
    assert logged["val/ERLE_oracle_count"] == pytest.approx(0.0)
    assert "val/ERLE_oracle" not in logged


@pytest.mark.unit
def test_trainer_erle_logs_oracle_only_with_explicit_echo_support():
    trainer = _erle_logging_trainer()
    stats = {
        "erle_db_sum": 5.0,
        "erle_count": 1,
        "oracle_erle_db_sum": 7.0,
        "oracle_erle_count": 1,
    }

    trainer._compute_and_log_erle(epoch=2, erle_stats=stats)

    logged = {name: value for name, value, _step in trainer.logged}
    assert logged["val/ERLE_oracle"] == pytest.approx(7.0)
    assert logged["val/ERLE_oracle_count"] == pytest.approx(1.0)


@pytest.mark.unit
def test_erle_worker_computes_oracle_only_when_echo_stem_is_explicit():
    from tasks.aec.trainer.aec_trainer import _compute_erle_one

    rng = np.random.default_rng(11)
    far = rng.normal(size=400).astype(np.float32)
    echo = far * 0.5
    near = np.zeros_like(echo)
    estimate = echo * 0.1
    without_oracle = _compute_erle_one((echo, near, estimate, far, None, 1000))
    with_oracle = _compute_erle_one((echo, near, estimate, far, echo, 1000))

    assert without_oracle[0] is not None
    assert without_oracle[1] is None
    assert with_oracle[0] == with_oracle[1]
    assert with_oracle[1] is not None
    assert with_oracle[1][0] == pytest.approx(20.0, abs=0.1)


@pytest.mark.unit
def test_trainer_erle_ddp_uses_one_fixed_shape_collective(monkeypatch):
    trainer = _erle_logging_trainer(world_size=2)
    calls = []

    def fake_all_reduce(tensor, op=None):
        calls.append((tensor.shape, tensor.dtype, op))
        tensor.add_(
            torch.tensor(
                [3.0, 1.0, 1.0, 30.0, 3.0, 10.0, 0.0, 0.0, 0.0],
                dtype=tensor.dtype,
            )
        )

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.dist.all_reduce", fake_all_reduce)
    trainer._compute_and_log_erle(
        epoch=1,
        erle_stats={
            "erle_db_sum": 1.0,
            "erle_count": 2,
            "erle_energy_sum_in": 10.0,
            "erle_energy_sum_out": 2.0,
            "erle_energy_count": 5,
        },
    )

    assert len(calls) == 1
    assert calls[0][0] == torch.Size([9])
    assert calls[0][1] == torch.float64
    logged = {name: value for name, value, _step in trainer.logged}
    assert logged["val/ERLE"] == pytest.approx(4.0 / 3.0)
    assert logged["val/ERLE_energy_micro"] == pytest.approx(10.0 * np.log10(40.0 / 5.0))


def _erle_val_ctx():
    return {
        "data": {},
        "state": None,
        "mic": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "farend": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        "nearend": torch.zeros(2, 2),
        "nearend_estimate": torch.zeros(2, 2),
        "lengths": torch.tensor([2, 2]),
        "fs_val": 16000,
        "has_echo": torch.tensor([True, False]),
        "is_pure_echo": torch.tensor([True, False]),
        "is_pure_echo_noise": torch.tensor([False, False]),
    }


def _erle_ddp_worker(rank: int, world_size: int, init_file: str, result_dir: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    try:
        import tasks.aec.trainer.aec_trainer as trainer_module

        trainer = object.__new__(AECTrainer)
        trainer.device = torch.device("cpu")
        trainer.rank = rank
        trainer.world_size = world_size
        trainer._aec_do_metrics = True
        trainer._vm_max_erle = 2
        trainer._vm_metrics_workers = 2
        extra = {
            "erle_db_sum": 0.0,
            "erle_count": 0,
            "erle_errors": [],
            "erle_remaining": 2 if rank == 0 else 0,
            "state_metrics_available": False,
        }
        ctx = _erle_val_ctx()
        ctx["has_echo"] = torch.tensor([True, True]) if rank == 0 else torch.tensor([False, False])
        ctx["is_pure_echo"] = ctx["has_echo"].clone()
        if rank == 0:
            def fail_compute(*_args, **_kwargs):
                raise RuntimeError("rank0 ERLE failure")

            class _ShutdownFailureExecutor(_ImmediateExecutor):
                def shutdown(self, wait=True, cancel_futures=False):
                    del wait, cancel_futures
                    raise OSError("rank0 executor shutdown failure")

            trainer_module.compute_erle = fail_compute
            trainer_module.ProcessPoolExecutor = _ShutdownFailureExecutor
            trainer_module.wait = _wait_all

        try:
            with trainer._validation_epoch_resources():
                trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)
                trainer._val_run_extra_metrics(epoch=1, extra=extra, do_metrics=True, val_x=1)
        except RuntimeError as exc:
            Path(result_dir, f"rank{rank}.txt").write_text(str(exc))
        except Exception as exc:
            Path(result_dir, f"rank{rank}.txt").write_text(
                f"wrong exception: {type(exc).__name__}: {exc}"
            )
        else:
            Path(result_dir, f"rank{rank}.txt").write_text("missing ERLE error")
    finally:
        dist.destroy_process_group()


def _erle_shutdown_ddp_worker(
    rank: int, world_size: int, init_file: str, result_dir: str
):
    """仅 rank 0 在 ERLE future 等待期收到信号，所有 rank 一致回传 aborted。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    try:
        import tasks.aec.trainer.aec_trainer as trainer_module

        trainer = object.__new__(AECTrainer)
        trainer.device = torch.device("cpu")
        trainer.rank = rank
        trainer.world_size = world_size
        trainer._shutdown_requested = False
        trainer._shutdown_signal = None
        trainer._aec_do_metrics = True
        trainer._vm_max_erle = 2
        trainer._vm_metrics_workers = 2
        trainer._log_scalar = lambda *_args, **_kwargs: None
        extra = {
            "erle_db_sum": 0.0,
            "erle_count": 0,
            "erle_errors": [],
            "erle_remaining": 2,
            "state_metrics_available": False,
        }
        ctx = _erle_val_ctx()
        ctx["has_echo"] = torch.tensor([True, True])
        ctx["is_pure_echo"] = torch.tensor([True, True])

        class _PendingFuture:
            def result(self):
                raise AssertionError("pending ERLE future must not be consumed")

        class _RankExecutor(_ImmediateExecutor):
            def submit(self, fn, task):
                if rank == 0:
                    return _PendingFuture()
                return super().submit(fn, task)

        def rank_wait(pending, **_kwargs):
            if rank == 0:
                trainer.request_shutdown(15)
                return set(), set(pending)
            return set(pending), set()

        trainer_module.ProcessPoolExecutor = _RankExecutor
        trainer_module.wait = rank_wait
        trainer_module.compute_erle = lambda *_args, **_kwargs: 3.0

        with trainer._validation_epoch_resources():
            trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)
            aborted = trainer._val_run_extra_metrics(
                epoch=1, extra=extra, do_metrics=True, val_x=1
            )

        assert aborted is True
        assert trainer.shutdown_requested()
        assert trainer._shutdown_signal == 15
        assert extra["erle_db_sum"] == 0.0
        assert extra["erle_count"] == 0
        assert extra["erle_errors"] == []
        assert extra["erle_result"] is None
        dist.barrier()
        Path(result_dir, f"shutdown-rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _spawn_erle_ddp(tmpdir: str, worker=_erle_ddp_worker) -> None:
    world_size = 2
    ctx = mp.spawn(
        worker,
        args=(world_size, str(Path(tmpdir) / "init_store"), tmpdir),
        nprocs=world_size,
        join=False,
    )
    deadline = time.monotonic() + 30
    while not ctx.join(max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in ctx.processes:
                if process.is_alive():
                    process.terminate()
            raise TimeoutError("ERLE DDP worker hung before the fixed collective")


@pytest.mark.integration
@pytest.mark.ddp
@pytest.mark.subprocess
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_erle_ddp_rank_local_failure_and_empty_echo_quota_raise_without_hang(tmp_path):
    _spawn_erle_ddp(str(tmp_path))

    assert "rank0 ERLE failure" in (tmp_path / "rank0.txt").read_text()
    assert "another rank failed" in (tmp_path / "rank1.txt").read_text()


@pytest.mark.integration
@pytest.mark.ddp
@pytest.mark.subprocess
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_erle_ddp_shutdown_wait_returns_aborted_on_all_ranks(tmp_path):
    _spawn_erle_ddp(str(tmp_path), worker=_erle_shutdown_ddp_worker)

    assert (tmp_path / "shutdown-rank0.ok").exists()
    assert (tmp_path / "shutdown-rank1.ok").exists()


@pytest.mark.unit
def test_val_extra_skips_erle_cpu_work_when_metrics_not_due(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = False
    extra = {"erle_db_sum": 0.0, "erle_count": 0, "state_metrics_available": False}

    def fail_compute(*_args, **_kwargs):
        raise AssertionError("compute_erle must not run")

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.compute_erle", fail_compute)
    trainer._val_accumulate_extra(extra, _erle_val_ctx(), is_ddp_pad=False)

    assert extra == {"erle_db_sum": 0.0, "erle_count": 0, "state_metrics_available": False}


@pytest.mark.unit
def test_val_extra_accumulates_per_utterance_db_only_on_metrics_round(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    extra = {"erle_db_sum": 0.0, "erle_count": 0, "state_metrics_available": False}
    calls = []

    def fake_erle(mic, nearend, enhanced, **kwargs):
        calls.append((mic.copy(), nearend.copy(), enhanced.copy(), kwargs))
        return 7.5

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.wait", _wait_all)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.compute_erle", fake_erle)
    ctx = _erle_val_ctx()
    trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], np.array([1.0, 2.0], dtype=np.float32))
    assert calls[0][3]["fs"] == 16000
    assert extra["erle_db_sum"] == 7.5
    assert extra["erle_count"] == 1

    trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=True)
    assert len(calls) == 1


@pytest.mark.unit
def test_val_extra_excludes_echo_scenes_without_clean_residual_proof(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 2
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_eligible_count": 0,
        "erle_remaining": 2,
        "state_metrics_available": False,
    }
    ctx = _erle_val_ctx()
    ctx["has_echo"] = torch.tensor([True, True])
    # canonical flags 无法区分 prob_target_echo 与 prob_target_noise_echo，必须排除。
    ctx["is_pure_echo"] = torch.tensor([False, False])
    ctx["is_pure_echo_noise"] = torch.tensor([False, True])
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.ProcessPoolExecutor",
        lambda *_args, **_kwargs: pytest.fail("polluted ERLE rows must not run"),
    )

    trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)

    assert extra["erle_count"] == 0
    assert extra["erle_eligible_count"] == 0
    assert extra["erle_remaining"] == 2


@pytest.mark.unit
def test_val_extra_prefers_explicit_echo_stem_over_residual_approximation(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 1
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_eligible_count": 0,
        "erle_remaining": 1,
        "state_metrics_available": False,
    }
    ctx = _erle_val_ctx()
    ctx["is_pure_echo"] = torch.tensor([False, False])
    ctx["echo"] = torch.tensor([[0.25, 0.5], [0.0, 0.0]])
    calls = []

    def fake_erle(observation, clean, enhanced, **kwargs):
        calls.append((observation.copy(), clean.copy(), enhanced.copy(), kwargs))
        return 6.0

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.wait", _wait_all)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.compute_erle", fake_erle)

    trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], np.array([0.25, 0.5], dtype=np.float32))
    assert np.array_equal(calls[0][1], np.zeros(2, dtype=np.float32))
    assert extra["erle_count"] == 1
    assert extra["oracle_erle_count"] == 1


@pytest.mark.unit
def test_val_extra_enforces_independent_erle_quota(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 1
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_remaining": 1,
        "state_metrics_available": False,
    }
    calls = []
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.wait", _wait_all)
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.compute_erle",
        lambda *_args, **_kwargs: calls.append(1) or 4.0,
    )
    ctx = _erle_val_ctx()
    ctx["has_echo"] = torch.tensor([True, True])
    ctx["is_pure_echo"] = torch.tensor([True, True])

    trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)
    trainer._val_accumulate_extra(extra, _erle_val_ctx(), is_ddp_pad=False)

    assert calls == [1]
    assert extra["erle_count"] == 1
    assert extra["erle_remaining"] == 0


@pytest.mark.unit
def test_validation_epoch_reuses_one_erle_executor_across_batches(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 4
    trainer._vm_metrics_workers = 2
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_remaining": 4,
        "state_metrics_available": False,
    }
    instances = []

    class _Executor(_ImmediateExecutor):
        def __init__(self, *, max_workers, mp_context):
            super().__init__(max_workers=max_workers, mp_context=mp_context)
            instances.append(self)

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.ProcessPoolExecutor", _Executor)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.wait", _wait_all)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.compute_erle", lambda *_a, **_k: 3.0)
    first = _erle_val_ctx()
    second = _erle_val_ctx()
    first["has_echo"] = second["has_echo"] = torch.tensor([True, True])
    first["is_pure_echo"] = second["is_pure_echo"] = torch.tensor([True, True])

    with trainer._validation_epoch_resources():
        trainer._val_accumulate_extra(extra, first, is_ddp_pad=False)
        trainer._val_accumulate_extra(extra, second, is_ddp_pad=False)
        assert trainer._erle_executor is instances[0]

    assert len(instances) == 1
    assert instances[0].shutdown_calls == [(False, False)]
    assert "_erle_executor" not in trainer.__dict__


@pytest.mark.unit
def test_erle_executor_capacity_is_not_locked_by_sparse_first_batch(monkeypatch):
    """首批仅一个 echo task 时，整轮复用池仍按配置/quota 创建容量。"""
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 6
    trainer._vm_metrics_workers = 4
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_remaining": 6,
        "state_metrics_available": False,
    }
    instances = []

    class _Executor(_ImmediateExecutor):
        def __init__(self, *, max_workers, mp_context):
            super().__init__(max_workers=max_workers, mp_context=mp_context)
            instances.append(self)

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.ProcessPoolExecutor", _Executor)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.wait", _wait_all)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.compute_erle", lambda *_a, **_k: 3.0)
    first = _erle_val_ctx()
    first["has_echo"] = torch.tensor([True, False])
    first["is_pure_echo"] = torch.tensor([True, False])
    second = _erle_val_ctx()
    second["has_echo"] = torch.tensor([True, True])
    second["is_pure_echo"] = torch.tensor([True, True])

    with trainer._validation_epoch_resources():
        trainer._val_accumulate_extra(extra, first, is_ddp_pad=False)
        trainer._val_accumulate_extra(extra, second, is_ddp_pad=False)

    assert len(instances) == 1
    assert instances[0].max_workers == min(4, os.cpu_count() or 1, 6)
    assert extra["erle_count"] == 3


@pytest.mark.unit
def test_erle_executor_shuts_down_once_when_metric_worker_raises(monkeypatch):
    """compute_erle 异常先记录，固定 ERLE collective 后统一抛出并释放进程池。"""
    trainer = object.__new__(AECTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.world_size = 1
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 2
    trainer._vm_metrics_workers = 2
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_errors": [],
        "erle_remaining": 2,
        "state_metrics_available": False,
    }
    instances = []

    class _Executor(_ImmediateExecutor):
        def __init__(self, *, max_workers, mp_context):
            super().__init__(max_workers=max_workers, mp_context=mp_context)
            instances.append(self)

    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.ProcessPoolExecutor", _Executor)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.wait", _wait_all)
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.compute_erle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected ERLE failure")
        ),
    )
    ctx = _erle_val_ctx()
    ctx["has_echo"] = torch.tensor([True, True])
    ctx["is_pure_echo"] = torch.tensor([True, True])

    with trainer._validation_epoch_resources():
        trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)
        assert len(extra["erle_errors"]) == 2
        assert all(
            error == "RuntimeError: injected ERLE failure"
            for error in extra["erle_errors"]
        )

    with pytest.raises(RuntimeError, match="injected ERLE failure"):
        trainer._compute_and_log_erle(epoch=1, erle_stats=extra)

    assert len(instances) == 1
    assert instances[0].shutdown_calls == [(False, False)]
    assert "_erle_executor" not in trainer.__dict__


@pytest.mark.integration
@pytest.mark.subprocess
def test_erle_spawn_pool_computes_real_metric():
    """真实 spawn pool 必须能 pickle worker 并有界关闭，而非只靠替身测试。"""
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 1
    trainer._vm_metrics_workers = 1
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_errors": [],
        "erle_remaining": 1,
        "state_metrics_available": False,
    }
    ctx = _erle_val_ctx()
    ctx["nearend_estimate"] = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    with trainer._validation_epoch_resources():
        trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)

    assert extra["erle_errors"] == []
    assert extra["erle_count"] == 1
    assert extra["erle_db_sum"] > 0
    assert "_erle_executor" not in trainer.__dict__


@pytest.mark.unit
def test_erle_shutdown_abort_is_not_escalated_to_runtime_error(monkeypatch):
    """信号中断 ERLE 仅 terminate 进程池并丢弃部分结果，不得升级为 RuntimeError。"""
    trainer = object.__new__(AECTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.world_size = 1
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 2
    trainer._vm_metrics_workers = 2
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_errors": [],
        "erle_remaining": 2,
        "state_metrics_available": False,
    }
    instances = []

    class _PendingFuture:
        def result(self):
            raise AssertionError("pending future must never be awaited")

    class _Executor(_ImmediateExecutor):
        def __init__(self, *, max_workers, mp_context):
            super().__init__(max_workers=max_workers, mp_context=mp_context)
            instances.append(self)

        def submit(self, fn, task):
            return _PendingFuture()

    # 外层 guard 查询时尚未停机，进入等待循环后信号到达。
    shutdown_polls = iter([False, True])
    trainer.shutdown_requested = lambda: next(shutdown_polls, True)
    monkeypatch.setattr("tasks.aec.trainer.aec_trainer.ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.wait",
        lambda pending, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wait must not run after shutdown is observed")
        ),
    )
    ctx = _erle_val_ctx()
    ctx["has_echo"] = torch.tensor([True, True])
    ctx["is_pure_echo"] = torch.tensor([True, True])

    with trainer._validation_epoch_resources():
        trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)
        # 主动停机不产生 ERLE 错误，部分结果在固定 collective 后统一丢弃。
        assert extra["erle_errors"] == []
        assert extra["erle_count"] == 0
        # 模拟停机前较早 batch 留下的历史错误；统一丢弃 partial epoch 时也必须清空。
        extra["erle_errors"] = ["RuntimeError: stale ERLE failure"]
        trainer._shutdown_requested = True
        trainer._shutdown_signal = 15
        aborted = trainer._val_run_extra_metrics(
            epoch=1, extra=extra, do_metrics=True, val_x=1
        )

    assert aborted is True
    assert extra["erle_errors"] == []
    assert instances[0].shutdown_calls == [(False, True)]
    assert "_erle_executor" not in trainer.__dict__


@pytest.mark.unit
def test_zero_erle_quota_skips_waveform_materialization(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_erle = 0
    extra = {
        "erle_db_sum": 0.0,
        "erle_count": 0,
        "erle_remaining": 0,
        "state_metrics_available": False,
    }
    monkeypatch.setattr(
        trainer,
        "_materialize_val_waveforms",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("waveforms must not materialize at zero quota")
        ),
    )
    monkeypatch.setattr(
        "tasks.aec.trainer.aec_trainer.compute_erle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ERLE must not run at zero quota")
        ),
    )

    trainer._val_accumulate_extra(extra, _erle_val_ctx(), is_ddp_pad=False)

    assert extra["erle_count"] == 0
    assert extra["erle_remaining"] == 0


@pytest.mark.unit
def test_val_metric_pairs_reuse_erle_waveform_cpu_cache():
    trainer = object.__new__(AECTrainer)
    trainer._aec_do_metrics = True
    trainer._vm_max_metrics = 2
    extra = {"erle_db_sum": 0.0, "erle_count": 0, "state_metrics_available": False}
    ctx = _erle_val_ctx()
    ctx["nearend"] = torch.tensor([[0.25, 0.5], [0.0, 0.0]])
    trainer._val_accumulate_extra(extra, ctx, is_ddp_pad=False)
    assert "waveforms_cpu" in ctx
    cached = ctx["waveforms_cpu"]
    pairs = []
    trainer._collect_val_metric_pairs(
        ctx["nearend_estimate"], ctx["nearend"], ctx["lengths"], 16000,
        pairs, 0, extra, ctx,
    )
    assert ctx["waveforms_cpu"] is cached
    assert np.shares_memory(pairs[0][0], cached["nearend"])
    assert np.shares_memory(pairs[0][1], cached["estimate"])


@pytest.mark.unit
def test_aec_shared_metric_path_excludes_erle(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer._vm_metrics = ["SI-SDR", "ERLE", "ESTOI"]
    trainer._vm_parallel_intrusive = False
    trainer._vm_metrics_workers = 0
    trainer.device = torch.device("cpu")
    trainer.shutdown_requested = lambda: False
    observed = []

    monkeypatch.setattr(
        "core.metrics.compute_metrics_batch",
        lambda _pairs, *, metrics, **_kwargs: observed.append(list(metrics)) or {},
    )

    trainer._merge_metrics_chunk([], {})

    assert observed == [["SI-SDR", "ESTOI"]]


@pytest.mark.unit
def test_erle_runtime_contract_applies_strict_coverage_and_history():
    trainer = _erle_logging_trainer()
    trainer._vm_metrics = ["SI-SDR", "ERLE"]
    trainer._vm_strict = False
    trainer._vm_min_coverage = {"ERLE": 0.75}
    trainer.val_metrics_epochs = [3]
    trainer.val_metrics_history = {"SI-SDR": [4.0]}
    trainer.val_metrics_coverage_history = {
        "SI-SDR": [
            {"eligible_count": 4.0, "valid_count": 4.0, "coverage_ratio": 1.0}
        ]
    }
    stats = {
        "erle_db_sum": 12.0,
        "erle_count": 2,
        "erle_eligible_count": 4,
    }

    with pytest.raises(RuntimeError, match="ERLE validation coverage.*below"):
        trainer._compute_and_log_erle(epoch=3, erle_stats=stats)

    assert "ERLE" not in trainer.val_metrics_history
    assert "ERLE" not in trainer.val_metrics_coverage_history
    assert not any(name.startswith("val/ERLE") for name, _value, _step in trainer.logged)


@pytest.mark.unit
def test_erle_task_result_uses_unified_logs_and_history():
    trainer = _erle_logging_trainer()
    trainer._vm_metrics = ["SI-SDR", "ERLE"]
    trainer._vm_strict = False
    trainer._vm_min_coverage = {}
    trainer._vm_enabled = True
    trainer._vm_parallel_intrusive = False
    trainer._vm_metrics_workers = 0
    trainer._nccl_flow_log = lambda *_args: None
    trainer._allreduce_sum = lambda value, **_kwargs: value
    trainer._comp_keys = []
    trainer.val_metrics_epochs = []
    trainer.val_metrics_history = {}
    trainer.val_metrics_coverage_history = {}
    trainer.val_metrics_loss_comp_hist = {}
    trainer._should_save_curve = lambda _count: False

    trainer._log_val_metrics(
        None,
        epoch=5,
        aggregated_intrusive={"SI-SDR": 8.0, "_count_SI-SDR": 2.0},
        eligible_count=2,
        task_metric_results={"ERLE": (6.0, 1.0, 2.0)},
    )

    logged = {name: value for name, value, step in trainer.logged if step == 5}
    assert logged["val/ERLE"] == pytest.approx(6.0)
    assert logged["val/ERLE/eligible_count"] == pytest.approx(2.0)
    assert logged["val/ERLE/valid_count"] == pytest.approx(1.0)
    assert logged["val/ERLE/coverage_ratio"] == pytest.approx(0.5)
    assert trainer.val_metrics_epochs == [5]
    assert trainer.val_metrics_history == {"SI-SDR": [4.0], "ERLE": [6.0]}
    assert trainer.val_metrics_coverage_history["ERLE"] == [
        {"eligible_count": 2.0, "valid_count": 1.0, "coverage_ratio": 0.5}
    ]


@pytest.mark.unit
def test_erle_zero_count_obeys_strict_runtime_contract():
    trainer = _erle_logging_trainer()
    trainer._vm_metrics = ["ERLE"]
    trainer._vm_strict = True
    trainer._vm_min_coverage = {}

    with pytest.raises(RuntimeError, match="strict=True.*ERLE"):
        trainer._compute_and_log_erle(
            epoch=1,
            erle_stats={"erle_count": 0, "erle_eligible_count": 2},
        )


@pytest.mark.unit
def test_erle_collective_is_gated_by_same_do_metrics_flag(monkeypatch):
    trainer = object.__new__(AECTrainer)
    trainer.world_size = 1
    calls = []
    trainer._compute_and_log_erle = lambda *args, **kwargs: calls.append((args, kwargs))
    stats = {"erle_db_sum": 0.0, "erle_count": 0}

    trainer._val_run_extra_metrics(epoch=1, extra=stats, do_metrics=False, val_x=1)
    trainer._val_run_extra_metrics(epoch=2, extra=stats, do_metrics=True, val_x=2)

    assert len(calls) == 1
    assert calls[0][0] == (2, stats)
    assert calls[0][1] == {
        "val_x": 2,
        "log_result": False,
        "metrics_aborted": False,
    }


@pytest.mark.unit
@pytest.mark.parametrize("policy", ["skip", "log_only"])
def test_offline_alignment_rejects_removed_pad_policies(policy):
    with pytest.raises(ValueError, match="only supports 'apply'"):
        align_farend_to_mic(
            np.zeros(32, dtype=np.float32),
            np.ones(32, dtype=np.float32),
            fs=100,
            pad_policy=policy,
        )


@pytest.mark.unit
def test_laec_disk_cache_uses_same_directory_fsync_and_atomic_replace(tmp_path, monkeypatch):
    frontend = LinearAECFrontend(
        {"backend": None, "cache_dir": str(tmp_path), "cache_mem_max": 0}
    )
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    fsync_calls = []
    replace_calls = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    def record_replace(src, dst):
        replace_calls.append((src, dst, np.load(src).copy()))
        return real_replace(src, dst)

    monkeypatch.setattr("tasks.aec.data.linear_aec_frontend.os.fsync", record_fsync)
    monkeypatch.setattr("tasks.aec.data.linear_aec_frontend.os.replace", record_replace)
    frontend._cache_save("contract", arr)

    target = tmp_path / "contract.npy"
    assert fsync_calls
    assert len(replace_calls) == 1
    src, dst, staged = replace_calls[0]
    assert os.path.dirname(src) == str(tmp_path)
    assert dst == str(target)
    np.testing.assert_array_equal(staged, arr)
    np.testing.assert_array_equal(np.load(target), arr)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.unit
def test_laec_cache_replace_failure_preserves_existing_cache(tmp_path, monkeypatch):
    frontend = LinearAECFrontend(
        {"backend": None, "cache_dir": str(tmp_path), "cache_mem_max": 0}
    )
    target = tmp_path / "contract.npy"
    original = np.array([9.0], dtype=np.float32)
    np.save(target, original)

    def fail_replace(_src, _dst):
        raise OSError("simulated concurrent replace failure")

    monkeypatch.setattr("tasks.aec.data.linear_aec_frontend.os.replace", fail_replace)
    frontend._cache_save("contract", np.array([1.0], dtype=np.float32))

    np.testing.assert_array_equal(np.load(target), original)
    assert not list(tmp_path.glob("*.tmp"))
