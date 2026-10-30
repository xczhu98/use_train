"""验证 batch-loop 编排与任务适配器合同。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from core.utils.ddp_error import is_unrecoverable_cuda_exception


@dataclass(frozen=True)
class ValidationBatchResult:
    """任务适配器完成单 batch 前向后交给公共编排层的数据。"""

    loss: torch.Tensor
    enhanced: torch.Tensor
    clean: torch.Tensor
    lengths: torch.Tensor
    fs: Any
    context: Any = None


@dataclass(frozen=True)
class ValidationLoopResult:
    """公共验证循环的本地累计结果。"""

    loss_sum: torch.Tensor
    finite_loss_samples: int
    total_samples: int
    metric_pairs: list[tuple[np.ndarray, np.ndarray, int]]
    forward_seconds: float
    aborted: bool

    @property
    def sample_count(self) -> int:
        """兼容旧调用方：loss 均值分母即 finite loss 样本数。"""
        return self.finite_loss_samples


class ValidationLoopAdapter(Protocol):
    """任务侧验证适配器；不得在这些回调中改变 collective 序列。"""

    def prepare_batch(self, raw_batch: Any, epoch: int, loop_count: int) -> Any: ...

    def forward_batch(
        self, prepared_batch: Any, epoch: int, loop_count: int
    ) -> ValidationBatchResult: ...

    def loss_weight(self, batch: ValidationBatchResult) -> int: ...

    def after_loss(
        self,
        batch: ValidationBatchResult,
        *,
        is_ddp_padding: bool,
        loss_was_finite: bool,
        weight: int,
    ) -> None: ...

    def collect_metric_pairs(
        self,
        batch: ValidationBatchResult,
        pairs: list[tuple[np.ndarray, np.ndarray, int]],
        total_pairs: int,
    ) -> int: ...

    def on_max_batches(self, epoch: int, max_batches: int) -> None: ...


def _run_synced_phase(trainer: Any, phase: str, callback):
    """同步普通 validation 异常；不可恢复 CUDA 故障立即传播。"""
    result = None
    local_error: Exception | None = None
    try:
        result = callback()
    except Exception as exc:
        if is_unrecoverable_cuda_exception(exc):
            raise
        local_error = exc
    with torch.inference_mode(False):
        error_context = trainer._sync_train_batch_exception(
            local_error,
            phase=f"validation_{phase}_error",
        )
    if error_context is not None:
        raise RuntimeError(f"validation {phase} failed: {error_context}") from local_error
    if local_error is not None:
        raise local_error
    return result


def run_validation_batch_loop(
    trainer: Any,
    epoch: int,
    adapter: ValidationLoopAdapter,
    *,
    collect_metrics: bool,
) -> ValidationLoopResult:
    """执行公共验证循环，并在固定 batch boundary 同步停机请求。"""
    loss_sum = torch.zeros(1, device=trainer.device)
    finite_loss_samples = 0
    total_samples = 0
    loop_count = 0
    pairs: list[tuple[np.ndarray, np.ndarray, int]] = []
    total_pairs = 0
    aborted = False

    started = time.monotonic()
    with trainer._ema_context(), torch.inference_mode():
        for raw_batch in trainer._iter_loader_with_synced_fetch(
            trainer.val_loader, phase="validation_loader"
        ):
            # 所有 rank 在 forward 前执行相同 shutdown collective；信号若在上一
            # forward 中到达，则此处一起停止，严禁按本地 flag 单独 break。
            if trainer._sync_shutdown_signum():
                aborted = True
                break
            loop_count += 1
            is_ddp_padding = trainer._val_iter_is_ddp_padding(loop_count)
            prepare = getattr(adapter, "prepare_batch", None)
            prepared_batch = _run_synced_phase(
                trainer,
                "prepare",
                lambda: prepare(raw_batch, epoch, loop_count)
                if callable(prepare)
                else raw_batch,
            )
            batch = _run_synced_phase(
                trainer,
                "forward_loss",
                lambda: adapter.forward_batch(prepared_batch, epoch, loop_count),
            )
            weight = _run_synced_phase(
                trainer,
                "loss_weight",
                lambda: adapter.loss_weight(batch),
            )
            if not is_ddp_padding:
                total_samples += weight
            loss_sum, finite_loss_samples, loss_was_finite = _run_synced_phase(
                trainer,
                "loss_accumulation",
                lambda: trainer._accumulate_val_loss(
                    batch.loss,
                    loss_sum,
                    finite_loss_samples,
                    is_ddp_padding,
                    epoch,
                    loop_count,
                    weight=weight,
                ),
            )
            _run_synced_phase(
                trainer,
                "after_loss",
                lambda: adapter.after_loss(
                    batch,
                    is_ddp_padding=is_ddp_padding,
                    loss_was_finite=loss_was_finite,
                    weight=weight,
                ),
            )

            def collect_local_metrics():
                if (
                    collect_metrics
                    and not is_ddp_padding
                    and total_pairs < trainer._vm_max_metrics
                ):
                    return adapter.collect_metric_pairs(batch, pairs, total_pairs)
                return total_pairs

            total_pairs = _run_synced_phase(
                trainer,
                "metric_pair_collection",
                collect_local_metrics,
            )

            reached_max_batches = (
                trainer.max_val_batches > 0
                and loop_count >= trainer.max_val_batches
            )
            _run_synced_phase(
                trainer,
                "max_batch_hook",
                lambda: adapter.on_max_batches(epoch, trainer.max_val_batches)
                if reached_max_batches
                else None,
            )
            if reached_max_batches:
                break
        # 所有 rank 都必须执行 terminal boundary collective。即使某个 rank
        # 已在 batch boundary 发现停机，也执行这一次固定调用，避免另一 rank
        # 在 forward 末尾才收到信号时出现 collective 序列分叉。
        terminal_aborted = bool(trainer._sync_shutdown_signum())
        aborted = aborted or terminal_aborted

    if aborted:
        pairs = []
    # 专用验证器若直接复用公共 batch-loop，可由 metrics 汇总层消费该口径，
    # 无需复制 finite-loss history 门禁；消费方会立即 pop，避免跨轮残留。
    trainer._validation_loop_finite_loss_samples = finite_loss_samples
    return ValidationLoopResult(
        loss_sum=loss_sum,
        finite_loss_samples=finite_loss_samples,
        total_samples=total_samples,
        metric_pairs=pairs,
        forward_seconds=time.monotonic() - started,
        aborted=aborted,
    )
