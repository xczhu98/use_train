"""DDP collective 一致性冒烟测试（Gloo CPU，2 进程）。

覆盖两个历史挂死场景（修复前均会在 collective 处永久等待）：

(a) 某 rank 注入 NaN loss：旧代码用与模型无关的 detached zero leaf 做
    backward，该 rank 不注册 DDP gradient hooks，其他 rank 在 all-reduce
    处挂死。修复后（_graph_preserving_zero_loss）两个 rank 一致完成 step。
(b) 某 rank validation pairs 为空：旧代码空 rank 提前 return，跳过
    n_chunk / total_utts 两次 all-reduce，collective 序列与非空 rank 不一致。
    修复后所有 rank 执行完全相同的 collective 序列，聚合结果正确。
(c) 只有一个 rank 在指标 chunk 中收到停机请求：chunk 后固定 shutdown
    collective 将请求传播到所有 rank，再一致跳过剩余指标并完成聚合序列。

环境无 pytest-timeout 插件，超时守卫内建于测试：spawn join timeout +
gloo init_process_group timeout，挂死时显式失败而非永久等待。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from conftest import spawn_ddp_workers
from core.utils.registry import LOSS_REGISTRY
from tasks.se.data.batch import SEBatch, SEBatchItem
from tasks.se.trainer.se_trainer import SETrainer

_WORLD_SIZE = 2
_SPAWN_TIMEOUT_SEC = 90

_TEST_LOSS_NAMES = ("_RankNaNLoss", "_RankErrorLoss")


@pytest.fixture(autouse=True, scope="module")
def _unregister_test_losses():
    """模块结束后注销本文件注册的临时 loss，避免污染全局 LOSS_REGISTRY。

    ``Registry`` 目前没有公开的注销 API，测试直接操作内部映射。
    """
    yield
    for name in _TEST_LOSS_NAMES:
        LOSS_REGISTRY._registry.pop(name, None)


class _TinySEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, noisy: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.conv(noisy.unsqueeze(1)).squeeze(1)


class _RankNaNLoss(nn.Module):
    """指定 rank 的指定调用序号返回 NaN，其余为 MSE。

    默认在第 2 次调用（最后一个 batch）注入：旧代码下该 rank 会少做一次
    gradient all-reduce，collective 计数与其他 rank 不一致，必然报错或挂死
    （若注入中间 batch，旧代码只是 collective 错位配对、静默污染梯度，冒烟
    测试难以观测）。
    """

    def __init__(self, nan_rank: int = 1, nan_calls: frozenset[int] = frozenset({2})) -> None:
        super().__init__()
        self.nan_rank = nan_rank
        self.nan_calls = nan_calls
        self.calls = 0
        self.rank = int(os.environ.get("_TEST_RANK", "-1"))

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls += 1
        if self.rank == self.nan_rank and self.calls in self.nan_calls:
            return torch.full((), float("nan"), device=enhanced.device, requires_grad=True)
        return F.mse_loss(enhanced, clean)


class _RankErrorLoss(nn.Module):
    """指定 rank 的指定调用序号抛出结构异常，其余为 MSE。"""

    def __init__(
        self,
        error_rank: int = 1,
        error_calls: frozenset[int] = frozenset({2}),
    ) -> None:
        super().__init__()
        self.error_rank = error_rank
        self.error_calls = error_calls
        self.calls = 0
        self.rank = int(os.environ.get("_TEST_RANK", "-1"))

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        self.calls += 1
        if self.rank == self.error_rank and self.calls in self.error_calls:
            raise RuntimeError("rank-local structural failure")
        return F.mse_loss(enhanced, clean)


class _SyntheticSEDataset(Dataset):
    def __init__(self, n_samples: int = 4, sr: int = 16000, length: int = 1024) -> None:
        self.n_samples = n_samples
        self.sr = sr
        self.length = length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(42 + idx)
        clean = torch.randn(self.length) * 0.3
        noise = torch.randn(self.length) * 0.1
        return SEBatchItem(clean=clean, noisy=clean + noise, sr=self.sr, length=self.length)


def _collate(items):
    clean = torch.stack([it.clean for it in items])
    noisy = torch.stack([it.noisy for it in items])
    sr = torch.tensor([it.sr for it in items], dtype=torch.long)
    lengths = torch.tensor([it.length for it in items], dtype=torch.long)
    return SEBatch(clean=clean, noisy=noisy, sr=sr, lengths=lengths)


def _patch_ddp_for_cpu() -> None:
    """base_trainer 固定按 CUDA 方式传 device_ids；CPU/Gloo 下必须去掉。

    用子类而非包装函数，保持 ``isinstance(m, DDP)``（_raw_model 解包）有效。
    """
    import core.trainer.base_trainer as bt

    class _CpuDDP(bt.DDP):
        def __init__(self, module, **kw):
            kw.pop("device_ids", None)
            kw.pop("output_device", None)
            super().__init__(module, **kw)

    bt.DDP = _CpuDDP


def _build_trainer(
    rank: int,
    world_size: int,
    exp_dir: Path,
    loss_fn: nn.Module,
    n_samples: int = 4,
    val_strict: bool = False,
) -> SETrainer:
    config = {
        "model": {"type": "TinySEModel", "args": {}},
        "trainer": {
            "epochs": 1,
            "gradient_clip_value": 5.0,
            "accumulation_steps": 1,
            "validation_epoch_interval": 0,
            "log_interval": 1,
            "exp_path": str(exp_dir),
            "max_val_batches": 1,
            "strict_checks": False,
            "save_batch_interval": 0,
            "save_curve_interval": 0,
            "val_metrics": {
                "enabled": True,
                "metrics": ["SI-SDR"],
                "interval": 1,
                "max_utterances": 8,
                "metrics_chunk_size": 1,
                "parallel_intrusive": False,
                "strict": val_strict,
            },
        },
        "meta": {"seed": 42, "use_amp": False},
        "data": {"placeholder_num_samples": 1024},
    }
    torch.manual_seed(0)  # 两 rank 初始参数一致（DDP 也会从 rank0 广播）
    model = _TinySEModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = _SyntheticSEDataset(n_samples=n_samples)
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
    return SETrainer(
        rank=rank,
        world_size=world_size,
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
        loss_fn=loss_fn,
        config=config,
        train_loader=dl,
        val_loader=dl,
        sample_rate=16000,
    )


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    os.environ["_TEST_RANK"] = str(rank)
    # 强制 CPU：本测试必须在无 GPU 环境可跑
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=_SPAWN_TIMEOUT_SEC - 15),
    )


def _worker_nan_loss(rank: int, world_size: int, init_file: str, tmpdir: str) -> None:
    """(a) rank 1 首个 batch 注入 NaN loss：两个 rank 都必须一致完成 epoch。"""
    _init_gloo(rank, world_size, init_file)
    try:
        _patch_ddp_for_cpu()
        trainer = _build_trainer(
            rank, world_size, Path(tmpdir) / f"nan_rank{rank}", _RankNaNLoss(nan_rank=1)
        )

        epoch_loss = trainer._train_epoch(epoch=0)
        assert epoch_loss is not None

        # 跨 rank 一致性：n_steps 相同、参数有限且逐位一致（all-reduce 后同步更新）
        param_sq_sum = sum(float((p.detach() ** 2).sum()) for p in trainer.model.parameters())
        t = torch.tensor([param_sq_sum, float(trainer.n_steps)], dtype=torch.float64)
        gathered = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(gathered, t)
        assert all(int(g[1].item()) == 2 for g in gathered), f"n_steps mismatch: {gathered}"
        assert all(np.isfinite(float(g[0])) for g in gathered), "non-finite params after NaN step"
        assert abs(float(gathered[0][0]) - float(gathered[1][0])) < 1e-6 * max(1.0, float(gathered[0][0])), (
            f"params diverged across ranks: {gathered[0][0]} vs {gathered[1][0]}"
        )
        dist.barrier()
        (Path(tmpdir) / f"nan_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _worker_empty_val_pairs(rank: int, world_size: int, init_file: str, tmpdir: str) -> None:
    """(b) rank 1 validation pairs 为空：collective 序列一致、聚合结果正确。"""
    from core.metrics.intrusive import compute_sisdr

    _init_gloo(rank, world_size, init_file)
    try:
        _patch_ddp_for_cpu()
        trainer = _build_trainer(
            rank, world_size, Path(tmpdir) / f"val_rank{rank}", _RankNaNLoss(nan_rank=-1)
        )

        rng = np.random.default_rng(1234)
        ref_a = rng.standard_normal(4096).astype(np.float32)
        enh_a = (ref_a + 0.01 * rng.standard_normal(4096)).astype(np.float32)
        ref_b = rng.standard_normal(4096).astype(np.float32)
        enh_b = (ref_b + 0.05 * rng.standard_normal(4096)).astype(np.float32)
        pairs = [(ref_a, enh_a, 16000), (ref_b, enh_b, 16000)] if rank == 0 else []

        trainer._run_validation_metrics_from_pairs(
            epoch=1, pairs=pairs, do_metrics=True, val_x=1
        )
        dist.barrier()

        if rank == 0:
            hist = trainer.val_metrics_history.get("SI-SDR")
            assert hist, "rank 0 missing SI-SDR history after aggregation"
            expected = (compute_sisdr(ref_a, enh_a) + compute_sisdr(ref_b, enh_b)) / 2.0
            assert abs(hist[-1] - expected) < 1e-5, (
                f"SI-SDR mean mismatch: got {hist[-1]}, expected {expected}"
            )
        (Path(tmpdir) / f"val_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _worker_metrics_shutdown_sync(
    rank: int, world_size: int, init_file: str, tmpdir: str
) -> None:
    """(c) 仅 rank 0 在首个 metrics chunk 收到信号，所有 rank 一致退出。"""
    _init_gloo(rank, world_size, init_file)
    try:
        _patch_ddp_for_cpu()
        trainer = _build_trainer(
            rank,
            world_size,
            Path(tmpdir) / f"metrics_shutdown_rank{rank}",
            _RankNaNLoss(nan_rank=-1),
        )
        trainer._vm_strict = True
        calls = 0

        def _fake_metrics_batch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if rank == 0 and calls == 1:
                trainer.request_shutdown(15)
            return {"SI-SDR": 1.0, "_count_SI-SDR": 1.0}

        import core.metrics as metrics_mod

        metrics_mod.compute_metrics_batch = _fake_metrics_batch
        pair = (
            np.ones(32, dtype=np.float32),
            np.ones(32, dtype=np.float32),
            16000,
        )
        trainer._run_validation_metrics_from_pairs(
            epoch=1,
            pairs=[pair, pair],
            do_metrics=True,
            val_x=1,
        )

        assert calls == 1, f"rank {rank} executed {calls} chunks after shutdown"
        assert trainer.shutdown_requested()
        assert trainer._shutdown_signal == 15
        dist.barrier()
        (Path(tmpdir) / f"metrics_shutdown_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _worker_metrics_shutdown_zero_pairs_strict(
    rank: int, world_size: int, init_file: str, tmpdir: str
) -> None:
    """(e) 零 metrics chunk + strict：仅 rank 0 持信号时全 rank 一致中止且不报错。"""
    _init_gloo(rank, world_size, init_file)
    try:
        _patch_ddp_for_cpu()
        trainer = _build_trainer(
            rank,
            world_size,
            Path(tmpdir) / f"metrics_zero_rank{rank}",
            _RankNaNLoss(nan_rank=-1),
        )
        trainer._vm_strict = True
        if rank == 0:
            trainer.request_shutdown(15)

        # n_chunk_max == 0 时循环体不执行；汇总后的固定 shutdown collective 必须
        # 把单 rank 信号传播到所有 rank，strict 不得因主动停机升级为 RuntimeError。
        trainer._run_validation_metrics_from_pairs(
            epoch=1,
            pairs=[],
            do_metrics=True,
            val_x=1,
        )

        assert trainer.shutdown_requested(), f"rank {rank} missed propagated shutdown"
        assert trainer._shutdown_signal == 15
        dist.barrier()
        (Path(tmpdir) / f"metrics_zero_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


class _ShutdownInjectingLoss(nn.Module):
    """仅 ``signal_rank`` 在第 2 次调用时写入停机请求（模拟单 rank 收到信号）。"""

    def __init__(self, signal_rank: int) -> None:
        super().__init__()
        self.signal_rank = signal_rank
        self.calls = 0
        self.trainer = None
        self.rank = int(os.environ.get("_TEST_RANK", "-1"))

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls += 1
        if self.rank == self.signal_rank and self.calls == 2 and self.trainer is not None:
            self.trainer.request_shutdown(15)  # SIGTERM
        return F.mse_loss(enhanced, clean)


def _worker_shutdown_sync(rank: int, world_size: int, init_file: str, tmpdir: str) -> None:
    """(c) 只有 rank 0 收到停机信号：两个 rank 必须在同一 optimizer 边界退出。

    8 samples / batch 2 = 4 batches；信号在 rank 0 第 2 次 loss 调用（batch 1）
    注入，该 optimizer boundary 的固定 collective（MAX over signum）把退出判定
    同步给 rank 1；未提交窗口丢弃，两边 n_steps/cursor/参数一致且不挂死。
    """
    _init_gloo(rank, world_size, init_file)
    try:
        _patch_ddp_for_cpu()
        loss_fn = _ShutdownInjectingLoss(signal_rank=0)
        trainer = _build_trainer(
            rank, world_size, Path(tmpdir) / f"shutdown_rank{rank}", loss_fn, n_samples=8
        )
        loss_fn.trainer = trainer

        trainer._train_epoch(epoch=1)

        assert trainer.n_steps == 1, f"rank {rank} n_steps={trainer.n_steps}"
        assert trainer._train_epoch_completed is False
        assert trainer._last_completed_batch_cursor == 1
        # MAX collective 必须把停机请求传播到未收到信号的 rank。
        assert trainer.shutdown_requested()
        assert trainer._shutdown_signal == 15

        param_sq_sum = sum(float((p.detach() ** 2).sum()) for p in trainer.model.parameters())
        t = torch.tensor([param_sq_sum], dtype=torch.float64)
        gathered = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(gathered, t)
        assert abs(float(gathered[0][0]) - float(gathered[1][0])) < 1e-6 * max(1.0, float(gathered[0][0])), (
            f"params diverged across ranks: {gathered[0][0]} vs {gathered[1][0]}"
        )
        dist.barrier()
        (Path(tmpdir) / f"shutdown_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _worker_validation_shutdown_sync(
    rank: int, world_size: int, init_file: str, tmpdir: str
) -> None:
    """仅 rank 0 在不同 validation boundary 收信号，全 rank 一致返回 None。"""
    _init_gloo(rank, world_size, init_file)
    try:
        _patch_ddp_for_cpu()
        for signal_after, expected_forwards in ((0, 0), (1, 1), (2, 2)):
            trainer = _build_trainer(
                rank,
                world_size,
                Path(tmpdir) / f"val_shutdown_{signal_after}_rank{rank}",
                _RankNaNLoss(nan_rank=-1),
                n_samples=4,
            )
            trainer.max_val_batches = 0
            trainer._shutdown_requested = False
            trainer._shutdown_signal = None
            original_forward = trainer._val_forward_prepared
            forwards = 0

            def forward_with_signal(raw_batch, epoch, n_loops):
                nonlocal forwards
                forwards += 1
                result = original_forward(raw_batch, epoch, n_loops)
                if rank == 0 and forwards == signal_after:
                    trainer.request_shutdown(15)
                return result

            trainer._val_forward_prepared = forward_with_signal
            if rank == 0 and signal_after == 0:
                trainer.request_shutdown(15)

            result = trainer._validation_epoch(1)

            assert result is None
            assert forwards == expected_forwards, (
                f"rank {rank} signal_after={signal_after} forwards={forwards}"
            )
            assert trainer._vm_epoch_count == 0
            assert trainer.val_metrics_history == {}
            assert trainer.val_metrics_epochs == []
            dist.barrier()
        (Path(tmpdir) / f"val_shutdown_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def _spawn_and_join(worker, tmpdir: str, extra_args=()) -> None:
    """spawn 两个 Gloo 进程并在超时内 join；挂死时 terminate 子进程并失败。

    ``ProcessContext.join(timeout)`` 在任一进程退出时即返回，False 仅表示
    「还有进程未结束」，因此由 ``spawn_ddp_workers`` 循环调用直到全部 join。
    """
    spawn_ddp_workers(
        worker,
        args=(_WORLD_SIZE, str(Path(tmpdir) / "init_store"), tmpdir)
        + tuple(extra_args),
        nprocs=_WORLD_SIZE,
        timeout_sec=_SPAWN_TIMEOUT_SEC,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_nan_loss_no_hang_and_consistent():
    """(a) 一个 rank 注入 NaN loss，两个 rank 一致完成 step 且不挂死。"""
    tmpdir = tempfile.mkdtemp(prefix="ddp_nan_")
    try:
        _spawn_and_join(_worker_nan_loss, tmpdir)
        assert (Path(tmpdir) / "nan_rank0.ok").exists()
        assert (Path(tmpdir) / "nan_rank1.ok").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_empty_val_pairs_collective_consistent():
    """(b) 一个 rank validation pairs 为空，metrics 聚合不挂死且结果正确。"""
    tmpdir = tempfile.mkdtemp(prefix="ddp_val_")
    try:
        _spawn_and_join(_worker_empty_val_pairs, tmpdir)
        assert (Path(tmpdir) / "val_rank0.ok").exists()
        assert (Path(tmpdir) / "val_rank1.ok").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_metrics_shutdown_request_is_synchronized_between_chunks():
    """(c) 指标 chunk 中仅 rank 0 收到信号，所有 rank 一致跳过后续 chunk。"""
    tmpdir = tempfile.mkdtemp(prefix="ddp_metrics_shutdown_")
    try:
        _spawn_and_join(_worker_metrics_shutdown_sync, tmpdir)
        assert (Path(tmpdir) / "metrics_shutdown_rank0.ok").exists()
        assert (Path(tmpdir) / "metrics_shutdown_rank1.ok").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_metrics_shutdown_zero_pairs_strict_no_raise():
    """(e) 零 metrics chunk + strict：仅 rank 0 持信号，全 rank 一致中止不挂死。"""
    tmpdir = tempfile.mkdtemp(prefix="ddp_metrics_zero_")
    try:
        _spawn_and_join(_worker_metrics_shutdown_zero_pairs_strict, tmpdir)
        assert (Path(tmpdir) / "metrics_zero_rank0.ok").exists()
        assert (Path(tmpdir) / "metrics_zero_rank1.ok").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_shutdown_request_exits_same_batch_boundary():
    """(d) 只有 rank 0 收到信号，两个 rank 仍在同一 batch 边界退出。"""
    tmpdir = tempfile.mkdtemp(prefix="ddp_shutdown_")
    try:
        _spawn_and_join(_worker_shutdown_sync, tmpdir)
        assert (Path(tmpdir) / "shutdown_rank0.ok").exists()
        assert (Path(tmpdir) / "shutdown_rank1.ok").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_validation_shutdown_boundaries_are_consistent():
    """首批前、首批后及末批后停机都应全 rank 一致且无 partial history。"""
    tmpdir = tempfile.mkdtemp(prefix="ddp_val_shutdown_")
    try:
        _spawn_and_join(_worker_validation_shutdown_sync, tmpdir)
        assert (Path(tmpdir) / "val_shutdown_rank0.ok").exists()
        assert (Path(tmpdir) / "val_shutdown_rank1.ok").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.unit
def test_strict_finite_error_is_specific_and_nonstrict_is_unchanged():
    from core.losses.build import _StrictLossWrapper
    from core.losses.composite import CompositeLoss, StrictFiniteLossError
    from core.utils.registry import LOSS_REGISTRY

    rank = int(os.environ.get("_TEST_RANK", "-1"))
    nan_loss = _RankNaNLoss(nan_rank=rank, nan_calls=frozenset({1}))
    enhanced = torch.ones(1, 8)
    clean = torch.zeros_like(enhanced)
    lengths = torch.tensor([8])
    with pytest.raises(StrictFiniteLossError, match="loss output.*NaN/Inf"):
        _StrictLossWrapper(nan_loss)(enhanced, clean, lengths)

    finite_aec = {"mic": enhanced, "nearend": clean, "nearend_estimate": enhanced}
    finite_aec["mic"] = torch.full_like(enhanced, float("nan"))
    with pytest.raises(StrictFiniteLossError, match="'mic'.*NaN/Inf"):
        _StrictLossWrapper(nn.Identity())(finite_aec)

    if "_RankNaNLoss" not in LOSS_REGISTRY:
        LOSS_REGISTRY.register()(_RankNaNLoss)
    components = [
        {
            "type": "_RankNaNLoss",
            "args": {"nan_rank": rank, "nan_calls": frozenset({1})},
        }
    ]
    with pytest.raises(StrictFiniteLossError, match=r"component\[0\].*NaN/Inf"):
        CompositeLoss(components, strict=True)(enhanced, clean, lengths)
    assert torch.isnan(CompositeLoss(components, strict=False)(enhanced, clean, lengths))


@pytest.mark.unit
def test_strict_wrapper_does_not_swallow_other_runtime_errors():
    class _StructuralFailure(nn.Module):
        def forward(self, *_args, **_kwargs):
            raise RuntimeError("structural failure")

    from core.losses.build import _StrictLossWrapper

    with pytest.raises(RuntimeError, match="structural failure"):
        _StrictLossWrapper(_StructuralFailure())(torch.ones(1, 2), torch.zeros(1, 2))


def _worker_strict_finite_loss(
    rank: int,
    world_size: int,
    init_file: str,
    tmpdir: str,
    mode: str,
) -> None:
    """单 rank strict finite 失败后，所有 rank 应在相同边界一致报错。"""
    _init_gloo(rank, world_size, init_file)
    try:
        _patch_ddp_for_cpu()
        from core.losses.build import _StrictLossWrapper
        from core.losses.composite import CompositeLoss
        from core.utils.registry import LOSS_REGISTRY

        if mode.startswith("composite_"):
            if "_RankNaNLoss" not in LOSS_REGISTRY:
                LOSS_REGISTRY.register()(_RankNaNLoss)
            if "_RankErrorLoss" not in LOSS_REGISTRY:
                LOSS_REGISTRY.register()(_RankErrorLoss)
            suffix = mode.removeprefix("composite_")
            failure_kind, later_type = suffix.split("_", maxsplit=1)
            first_type = "_RankNaNLoss" if failure_kind == "nan" else "_RankErrorLoss"
            loss_fn = CompositeLoss(
                components=[
                    {
                        "type": first_type,
                        "args": {
                            "nan_calls" if failure_kind == "nan" else "error_calls": frozenset({1})
                        },
                    },
                    {"type": later_type},
                ],
                strict=True,
            )
        else:
            loss_fn = _StrictLossWrapper(_RankNaNLoss(nan_rank=1, nan_calls=frozenset({1})))
        trainer = _build_trainer(
            rank,
            world_size,
            Path(tmpdir) / f"strict_{mode}_rank{rank}",
            loss_fn,
        )
        trainer._strict_checks = True

        try:
            if mode == "single_val":
                trainer._validation_epoch(epoch=0)
            else:
                trainer._train_epoch(epoch=0)
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("strict finite failure was not reported")

        messages = [None] * world_size
        dist.all_gather_object(messages, message)
        assert messages[0] == messages[1]
        if mode.startswith("composite_error_"):
            assert "rank-local structural failure" in message, repr(message)
        else:
            assert "NaN/Inf" in message or "non-finite" in message, repr(message)
        dist.barrier()
        (Path(tmpdir) / f"strict_{mode}_rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
@pytest.mark.parametrize(
    "mode",
    [
        "single_train",
        "composite_nan_SISNRLoss",
        "composite_nan_GTCRNLogRatioLoss",
        "composite_error_SISNRLoss",
        "composite_error_GTCRNLogRatioLoss",
        "single_val",
    ],
)
def test_ddp_strict_finite_loss_is_synchronized_without_timeout(mode):
    """single wrapper、Composite 与 validation 均不得单 rank 提前抛错。"""
    tmpdir = tempfile.mkdtemp(prefix=f"ddp_strict_{mode}_")
    try:
        _spawn_and_join(_worker_strict_finite_loss, tmpdir, extra_args=(mode,))
        assert (Path(tmpdir) / f"strict_{mode}_rank0.ok").exists()
        assert (Path(tmpdir) / f"strict_{mode}_rank1.ok").exists()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_ddp_nan_loss_no_hang_and_consistent()
    test_ddp_empty_val_pairs_collective_consistent()
    test_ddp_metrics_shutdown_request_is_synchronized_between_chunks()
    test_ddp_metrics_shutdown_zero_pairs_strict_no_raise()
    test_ddp_shutdown_request_exits_same_batch_boundary()
    print("DDP collective consistency tests passed.")

pytestmark = [pytest.mark.integration, pytest.mark.ddp, pytest.mark.subprocess]
