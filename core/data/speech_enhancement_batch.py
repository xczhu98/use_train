"""SE 的 canonical Batch；共享 collate 与任务消费端使用同一类型。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from core.data.waveform_batch import (
    validate_batch_flag,
    validate_sample_rates,
    validate_waveform,
    validate_waveform_lengths,
)


@dataclass
class SEBatch:
    """collate 直接产出的 SE 批；clean 为 [B,T]，noisy 为 [B,T] 或 [B,C,T]。"""

    clean: torch.Tensor
    noisy: torch.Tensor
    sr: torch.Tensor
    lengths: torch.Tensor
    is_pure_noise: Optional[torch.Tensor] = None
    is_pure_echo: Optional[torch.Tensor] = None
    is_pure_echo_noise: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        """返回 clean/noisy 的 canonical 主样本数。"""
        if not isinstance(self.clean, torch.Tensor) or self.clean.ndim < 1:
            raise ValueError("SEBatch.clean must expose a batch dimension")
        return int(self.clean.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "SEBatch":
        """按公共 ``BatchContract`` 搬运 SE 的全部 tensor 字段。"""
        return replace(
            self,
            clean=self.clean.to(device=device, non_blocking=non_blocking),
            noisy=self.noisy.to(device=device, non_blocking=non_blocking),
            sr=self.sr.to(device=device, non_blocking=non_blocking),
            lengths=self.lengths.to(device=device, non_blocking=non_blocking),
            is_pure_noise=(
                None
                if self.is_pure_noise is None
                else self.is_pure_noise.to(device=device, non_blocking=non_blocking)
            ),
            is_pure_echo=(
                None
                if self.is_pure_echo is None
                else self.is_pure_echo.to(device=device, non_blocking=non_blocking)
            ),
            is_pure_echo_noise=(
                None
                if self.is_pure_echo_noise is None
                else self.is_pure_echo_noise.to(device=device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> "SEBatch":
        """锁页全部 tensor 字段，保留可选门控与 Batch 类型。"""
        return replace(
            self,
            clean=self.clean.pin_memory(),
            noisy=self.noisy.pin_memory(),
            sr=self.sr.pin_memory(),
            lengths=self.lengths.pin_memory(),
            is_pure_noise=(
                None if self.is_pure_noise is None else self.is_pure_noise.pin_memory()
            ),
            is_pure_echo=(
                None if self.is_pure_echo is None else self.is_pure_echo.pin_memory()
            ),
            is_pure_echo_noise=(
                None if self.is_pure_echo_noise is None else self.is_pure_echo_noise.pin_memory()
            ),
        )

    def validate(self) -> None:
        """校验布局、长度和门控；不改变现有波形有限值处理策略。"""
        size, time = validate_waveform(self.clean, name="SEBatch.clean")
        noisy_shape = validate_waveform(self.noisy, name="SEBatch.noisy", spatial=True)
        if noisy_shape != (size, time):
            raise ValueError("SEBatch clean/noisy batch and time dimensions must match")
        validate_waveform_lengths(
            self.lengths, batch_size=size, max_length=time, context="SEBatch"
        )
        validate_sample_rates(self.sr, batch_size=size, name="SEBatch.sr")
        for name in ("is_pure_noise", "is_pure_echo", "is_pure_echo_noise"):
            validate_batch_flag(getattr(self, name), batch_size=size, name=f"SEBatch.{name}")
