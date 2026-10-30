"""AEC 的结构化批合同：collate 直接产出 AECBatch，Trainer 按字段消费。"""

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
class AECBatchItem:
    """Single AEC sample returned by a Dataset (before collating)."""

    nearend: torch.Tensor
    mic: torch.Tensor
    farend: torch.Tensor
    fs: int
    length: int
    error: Optional[torch.Tensor] = None
    error_mask: Optional[torch.Tensor] = None
    is_pure_noise: bool = False
    is_pure_echo: bool = False
    is_pure_echo_noise: bool = False
    has_echo: bool = False
    state: Optional[torch.Tensor] = None


@dataclass
class AECBatch:
    """AEC 批：mic 为 [B,T] 或 [B,C,T]，其余波形为 [B,T]。

    flags/error 可省略；state 为可选 [B,T_state] int64，补齐值为 -1。
    """

    nearend: torch.Tensor
    mic: torch.Tensor
    farend: torch.Tensor
    fs: torch.Tensor
    lengths: torch.Tensor
    error: Optional[torch.Tensor] = None
    error_mask: Optional[torch.Tensor] = None
    is_pure_noise: Optional[torch.Tensor] = None
    is_pure_echo: Optional[torch.Tensor] = None
    is_pure_echo_noise: Optional[torch.Tensor] = None
    has_echo: Optional[torch.Tensor] = None
    state: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        """返回 nearend/mic 的 canonical 主样本数。"""
        if not isinstance(self.nearend, torch.Tensor) or self.nearend.ndim < 1:
            raise ValueError("AECBatch.nearend must expose a batch dimension")
        return int(self.nearend.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AECBatch":
        """按公共 ``BatchContract`` 搬运 AEC 的全部 tensor 字段。"""

        def _move(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return (
                None
                if value is None
                else value.to(device=device, non_blocking=non_blocking)
            )

        return replace(
            self,
            nearend=self.nearend.to(device=device, non_blocking=non_blocking),
            mic=self.mic.to(device=device, non_blocking=non_blocking),
            farend=self.farend.to(device=device, non_blocking=non_blocking),
            fs=self.fs.to(device=device, non_blocking=non_blocking),
            lengths=self.lengths.to(device=device, non_blocking=non_blocking),
            error=_move(self.error),
            error_mask=_move(self.error_mask),
            is_pure_noise=_move(self.is_pure_noise),
            is_pure_echo=_move(self.is_pure_echo),
            is_pure_echo_noise=_move(self.is_pure_echo_noise),
            has_echo=_move(self.has_echo),
            state=_move(self.state),
        )

    def pin_memory(self) -> "AECBatch":
        """锁页波形、残差、逐样本标志和可选 state。"""

        def _pin(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value.pin_memory()

        return replace(
            self,
            nearend=self.nearend.pin_memory(),
            mic=self.mic.pin_memory(),
            farend=self.farend.pin_memory(),
            fs=self.fs.pin_memory(),
            lengths=self.lengths.pin_memory(),
            error=_pin(self.error),
            error_mask=_pin(self.error_mask),
            is_pure_noise=_pin(self.is_pure_noise),
            is_pure_echo=_pin(self.is_pure_echo),
            is_pure_echo_noise=_pin(self.is_pure_echo_noise),
            has_echo=_pin(self.has_echo),
            state=_pin(self.state),
        )

    def validate(self) -> None:
        """校验完整结构；波形有限值扫描仍由 collate/Trainer 的策略控制。"""
        size, time = validate_waveform(self.nearend, name="AECBatch.nearend")
        for name in ("mic", "farend", "error"):
            value = getattr(self, name)
            if name == "error" and value is None:
                continue
            shape = validate_waveform(value, name=f"AECBatch.{name}", spatial=name == "mic")
            if shape != (size, time):
                raise ValueError(
                    f"AECBatch.{name} batch and time dimensions must match nearend"
                )
        validate_waveform_lengths(
            self.lengths, batch_size=size, max_length=time, context="AECBatch"
        )
        validate_sample_rates(self.fs, batch_size=size, name="AECBatch.fs")
        for name in (
            "error_mask", "is_pure_noise", "is_pure_echo", "is_pure_echo_noise", "has_echo"
        ):
            validate_batch_flag(getattr(self, name), batch_size=size, name=f"AECBatch.{name}")
        if self.error is None and self.error_mask is not None and bool(self.error_mask.any()):
            raise ValueError("AECBatch.error_mask cannot select a missing error waveform")
        if self.state is not None:
            if not isinstance(self.state, torch.Tensor) or self.state.dtype != torch.int64:
                raise TypeError("AECBatch.state must be an int64 tensor")
            if self.state.ndim != 2 or self.state.shape[0] != size or self.state.shape[1] < 1:
                raise ValueError("AECBatch.state must have non-empty shape [B,T_state]")
            if bool((self.state < -1).any()):
                raise ValueError("AECBatch.state must contain class indices or padding -1")
