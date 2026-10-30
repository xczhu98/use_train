"""Typed batch contract for Speech Separation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass
class SSBatch:
    """A fixed-source-count separation batch.

    Shapes:
        mixture: ``[B, T]`` (legacy) or ``[B, C, T]``
        sources: ``[B, K, T]``; targets remain reference-channel waveforms
        lengths: ``[B]``
        source_mask: ``[B, K]``; fixed-K batches must select every source
        sr: ``[B]`` or a scalar tensor
    """

    mixture: torch.Tensor
    sources: torch.Tensor
    lengths: torch.Tensor
    source_mask: torch.Tensor
    sr: torch.Tensor
    source_ids: tuple[tuple[str, ...], ...] | None = None

    @property
    def batch_size(self) -> int:
        """返回 mixture 的 canonical 主样本数。"""
        if not isinstance(self.mixture, torch.Tensor) or self.mixture.ndim < 1:
            raise ValueError("SSBatch.mixture must expose a batch dimension")
        return int(self.mixture.shape[0])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "SSBatch":
        """把 tensor 字段搬到目标设备，保留 source identity 元数据。"""
        return replace(
            self,
            mixture=self.mixture.to(device=device, non_blocking=non_blocking),
            sources=self.sources.to(device=device, non_blocking=non_blocking),
            lengths=self.lengths.to(device=device, non_blocking=non_blocking),
            source_mask=self.source_mask.to(device=device, non_blocking=non_blocking),
            sr=self.sr.to(device=device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "SSBatch":
        """参与 DataLoader 的递归锁页内存，保证后续 non_blocking 拷贝有效。"""
        return replace(
            self,
            mixture=self.mixture.pin_memory(),
            sources=self.sources.pin_memory(),
            lengths=self.lengths.pin_memory(),
            source_mask=self.source_mask.pin_memory(),
            sr=self.sr.pin_memory(),
        )

    def validate(self) -> None:
        """Validate the fixed-K source-axis contract before device transfer.

        ``sources`` always carries the configured source count, so every source mask
        entry must be active. Variable-K or partially masked samples are not part of
        the SS batch contract.
        """
        if not isinstance(self.mixture, torch.Tensor):
            raise TypeError("mixture must be a tensor")
        if self.mixture.ndim not in (2, 3):
            raise ValueError("mixture must have shape [B, T] or [B, C, T]")
        if not isinstance(self.sources, torch.Tensor):
            raise TypeError("sources must be a tensor")
        if self.sources.ndim != 3:
            raise ValueError("sources must have shape [B, K, T]")

        batch = int(self.mixture.shape[0])
        channels = 1 if self.mixture.ndim == 2 else int(self.mixture.shape[1])
        time = int(self.mixture.shape[-1])
        source_batch, source_count, source_time = map(int, self.sources.shape)
        if batch < 1:
            raise ValueError("mixture batch size B must be positive")
        if channels < 1:
            raise ValueError("mixture channel count C must be positive")
        if time < 1:
            raise ValueError("mixture time length T must be positive")
        if source_count < 1:
            raise ValueError("sources count K must be positive")
        if source_batch != batch or source_time != time:
            raise ValueError(
                "sources must have shape [B, K, T] aligned with mixture; "
                f"got mixture={tuple(self.mixture.shape)}, sources={tuple(self.sources.shape)}"
            )
        for name, value in (("mixture", self.mixture), ("sources", self.sources)):
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must use a floating-point dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")

        if not isinstance(self.lengths, torch.Tensor):
            raise TypeError("lengths must be a tensor")
        if self.lengths.shape != (batch,):
            raise ValueError(f"lengths must have shape [{batch}]")
        if self.lengths.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("lengths must use an integer dtype")
        if bool(((self.lengths < 1) | (self.lengths > time)).any()):
            raise ValueError(f"lengths must be in [1, {time}]")

        expected_mask = (batch, source_count)
        if not isinstance(self.source_mask, torch.Tensor):
            raise TypeError("source_mask must be a tensor")
        if tuple(self.source_mask.shape) != expected_mask:
            raise ValueError(
                f"source_mask must have shape {expected_mask}, "
                f"got {tuple(self.source_mask.shape)}"
            )
        if self.source_mask.dtype is not torch.bool:
            raise TypeError("source_mask must use torch.bool")
        if not bool(self.source_mask.all()):
            raise ValueError("source_mask must select every source in a fixed-K SS batch")

        if not isinstance(self.sr, torch.Tensor):
            raise TypeError("sr must be a tensor")
        if self.sr.ndim not in (0, 1) or (
            self.sr.ndim == 1 and self.sr.shape[0] != batch
        ):
            raise ValueError("sr must be scalar or have shape [B]")
        if self.sr.dtype is torch.bool or torch.is_complex(self.sr):
            raise TypeError("sr must use an integer or floating-point dtype")
        if not (torch.is_floating_point(self.sr) or self.sr.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }):
            raise TypeError("sr must use an integer or floating-point dtype")
        if not bool(torch.isfinite(self.sr).all()):
            raise ValueError("sr values must be finite")
        if bool((self.sr <= 0).any()):
            raise ValueError("sr values must be positive")

        if self.source_ids is not None:
            if len(self.source_ids) != batch:
                raise ValueError("source_ids must contain one row per batch item")
            for row in self.source_ids:
                if len(row) != source_count:
                    raise ValueError("source_ids rows must match source count K")
                if any(not isinstance(value, str) or not value for value in row):
                    raise ValueError("source_ids values must be non-empty strings")
