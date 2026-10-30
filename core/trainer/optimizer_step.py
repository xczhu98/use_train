"""Optimizer-step 的无状态 gate policy 与准备结果数据对象。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PreparedOptimizerStep:
    """已完成 unscale/sanitize/clip、尚未提交副作用的 step。"""

    key: str
    optimizer: torch.optim.Optimizer
    amp_fp16: bool
    eligible: bool
    grad_norm: float = 0.0
    reason: str | None = None
    # AMP 判定必须依赖显式标志：``reason`` 只是人类可读文案，改文案不得
    # 影响 GradScaler 的溢出判定（否则扩缩逻辑会静默失效）。
    amp_nonfinite: bool = False


def amp_input_is_finite(prepared: PreparedOptimizerStep) -> bool:
    """返回 prepare 阶段是否未发现 AMP non-finite 梯度。"""
    return not (prepared.amp_fp16 and prepared.amp_nonfinite)


def optimizer_gate_values(prepared: PreparedOptimizerStep) -> tuple[bool, bool]:
    """生成单 optimizer 跨 rank AND 所需的 eligible/AMP flags。"""
    return prepared.eligible, amp_input_is_finite(prepared)


def paired_optimizer_gate_values(
    prepared_steps: tuple[PreparedOptimizerStep, ...],
    *,
    window_eligible: bool,
) -> tuple[bool, bool]:
    """生成 GAN 原子 step pair 的本地 eligible/AMP flags。"""
    return (
        bool(window_eligible and all(step.eligible for step in prepared_steps)),
        all(amp_input_is_finite(step) for step in prepared_steps),
    )
