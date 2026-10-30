"""Validated dict-item batch contract for Target Speaker Extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from core.data.simulation.array_profiles import ResolvedMicrophoneArray
from core.data.simulation.trajectory import TrajectoryTimebase
from tasks.tse.cue_contract import CUE_SLOTS, TSECueBatches, TSECueSpecs


@dataclass
class TSEBatch:
    """Sample-major TSE batch produced by ``tse_collate_fn``.

    ``mixture`` 始终为 ``[B,C,T]``，``target`` 和全部 cue 为
    ``[R,...]``，其中 ``R=sum(num_targets)``。目标行顺序只由
    ``num_targets`` 推导，不维护第二份 target-to-mixture 索引。
    """

    mixture: torch.Tensor
    target: torch.Tensor
    sr: torch.Tensor
    lengths: torch.Tensor
    num_targets: torch.Tensor
    spk_labels: torch.Tensor
    target_present: torch.Tensor
    cues: TSECueBatches
    cue_specs: TSECueSpecs
    keys: tuple[str, ...]
    speaker_ids: tuple[str | None, ...]
    geometry: tuple[ResolvedMicrophoneArray | None, ...]
    timebase: tuple[TrajectoryTimebase | None, ...]
    provenance: tuple[Any | None, ...]

    @property
    def batch_size(self) -> int:
        """返回 mixture 的 canonical 主样本数。"""
        if not isinstance(self.mixture, torch.Tensor) or self.mixture.ndim < 1:
            raise ValueError("TSEBatch.mixture must expose a batch dimension")
        return int(self.mixture.shape[0])

    def validate(self) -> None:
        if not isinstance(self.mixture, torch.Tensor) or self.mixture.ndim != 3:
            raise ValueError("mixture must be a tensor with shape [B, C, T]")
        if not isinstance(self.target, torch.Tensor) or self.target.ndim != 2:
            raise ValueError("target must be a tensor with shape [R, T]")
        if self.mixture.dtype != torch.float32 or self.target.dtype != torch.float32:
            raise TypeError("mixture and target must use torch.float32")

        batch_size, channels, time = map(int, self.mixture.shape)
        rows, target_time = map(int, self.target.shape)
        if batch_size < 1 or channels < 1 or time < 1:
            raise ValueError("mixture B, C and T dimensions must be positive")
        if rows < 1 or target_time != time:
            raise ValueError("target must have shape [R, T] aligned with mixture time")

        if not isinstance(self.sr, torch.Tensor) or self.sr.ndim != 0:
            raise ValueError("sr must be a scalar tensor")
        if self.sr.dtype != torch.int32:
            raise TypeError("sr must use torch.int32")
        self._validate_vector(self.lengths, "lengths", batch_size, torch.int32)
        self._validate_vector(self.num_targets, "num_targets", batch_size, torch.int32)
        self._validate_vector(self.spk_labels, "spk_labels", rows, torch.int64)
        self._validate_vector(self.target_present, "target_present", rows, torch.bool)
        if not isinstance(self.cues, TSECueBatches):
            raise TypeError("cues must be TSECueBatches")
        if not isinstance(self.cue_specs, TSECueSpecs):
            raise TypeError("cue_specs must be TSECueSpecs")
        self._validate_tensor_devices()

        if not bool(torch.isfinite(self.mixture).all().item()):
            raise ValueError("mixture must contain only finite values")
        if not bool(torch.isfinite(self.target).all().item()):
            raise ValueError("target must contain only finite values")
        sample_rate = int(self.sr.item())
        if sample_rate <= 0:
            raise ValueError("sr must be positive")
        if bool(((self.lengths < 1) | (self.lengths > time)).any().item()):
            raise ValueError(f"lengths must be in [1, {time}]")
        if bool((self.num_targets < 1).any().item()):
            raise ValueError("num_targets must contain positive values")
        if int(self.num_targets.sum().item()) != rows:
            raise ValueError("sum(num_targets) must equal target row count R")

        target_lengths = torch.repeat_interleave(
            self.lengths, self.num_targets.to(torch.int64), output_size=rows,
        )
        times = torch.arange(time, device=self.mixture.device)
        target_nonzero = self.target != 0
        checks = (
            (target_nonzero & (times[None, :] >= target_lengths[:, None])).any(),
            (target_nonzero.any(dim=1) & ~self.target_present).any(),
            ((self.mixture != 0).any(dim=1) & (times[None, :] >= self.lengths[:, None])).any(),
        )
        messages = (
            "target samples after their valid length must be zero",
            "target_present=False requires zero target supervision",
            "mixture samples after their valid length must be zero",
        )
        for failed, message in zip(torch.stack(checks).detach().cpu().tolist(), messages):
            if failed:
                raise ValueError(message)
        self.cues.validate(self.cue_specs, rows)

        if len(self.keys) != batch_size or any(
            not isinstance(key, str) or not key for key in self.keys
        ):
            raise ValueError("keys must contain one non-empty string per mixture")
        if len(self.speaker_ids) != rows or any(
            value is not None and (not isinstance(value, str) or not value)
            for value in self.speaker_ids
        ):
            raise ValueError("speaker_ids must contain one string or None per target row")
        if len(self.geometry) != batch_size:
            raise ValueError("geometry must contain one host value per mixture")
        if len(self.timebase) != batch_size:
            raise ValueError("timebase must contain one host value per mixture")
        if len(self.provenance) != batch_size:
            raise ValueError("provenance must contain one host value per mixture")

        for geometry in self.geometry:
            if geometry is not None and not isinstance(geometry, ResolvedMicrophoneArray):
                raise TypeError("geometry values must be ResolvedMicrophoneArray or None")
        has_geometry = tuple(geometry is not None for geometry in self.geometry)
        if any(has_geometry) and not all(has_geometry):
            raise ValueError("geometry must be provided consistently for every batch item")
        if channels > 1 and not all(has_geometry):
            raise ValueError("multichannel TSE batches require resolved microphone geometry")
        resolved_geometry = tuple(
            geometry for geometry in self.geometry if geometry is not None
        )
        if resolved_geometry:
            if any(geometry.channels != channels for geometry in resolved_geometry):
                raise ValueError("geometry channel count must match mixture C")
            topology_keys = {
                (geometry.positions_m, geometry.reference_channel)
                for geometry in resolved_geometry
            }
            if len(topology_keys) != 1:
                raise ValueError("geometry topology and reference channel must match within a batch")

        valid_lengths = self.lengths.detach().cpu().tolist() if any(value is not None for value in self.timebase) else None
        for index, timebase in enumerate(self.timebase):
            if timebase is None:
                continue
            if not isinstance(timebase, TrajectoryTimebase):
                raise TypeError("timebase values must be TrajectoryTimebase or None")
            if timebase.sample_rate != sample_rate:
                raise ValueError("timebase sample_rate must match batch sr")
            if timebase.num_samples != valid_lengths[index]:
                raise ValueError("timebase num_samples must match the item valid length")

    def _validate_tensor_devices(self) -> None:
        tensors = [
            self.mixture,
            self.target,
            self.sr,
            self.lengths,
            self.num_targets,
            self.spk_labels,
            self.target_present,
        ]
        for slot in CUE_SLOTS:
            cue_batch = getattr(self.cues, slot)
            if cue_batch is None:
                continue
            for value in (
                cue_batch.value,
                cue_batch.present,
                cue_batch.lengths,
                cue_batch.timebase,
            ):
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all TSEBatch tensors, including cues, must be on one device")

    @staticmethod
    def _validate_vector(
        value: torch.Tensor,
        name: str,
        size: int,
        dtype: torch.dtype,
    ) -> None:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (size,):
            raise ValueError(f"{name} must have shape [{size}]")
        if value.dtype != dtype:
            raise TypeError(f"{name} must use {dtype}")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TSEBatch":
        return replace(
            self,
            mixture=self.mixture.to(device=device, non_blocking=non_blocking),
            target=self.target.to(device=device, non_blocking=non_blocking),
            sr=self.sr.to(device=device, non_blocking=non_blocking),
            lengths=self.lengths.to(device=device, non_blocking=non_blocking),
            num_targets=self.num_targets.to(device=device, non_blocking=non_blocking),
            spk_labels=self.spk_labels.to(device=device, non_blocking=non_blocking),
            target_present=self.target_present.to(
                device=device, non_blocking=non_blocking
            ),
            cues=self.cues.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "TSEBatch":
        return replace(
            self,
            mixture=self.mixture.pin_memory(),
            target=self.target.pin_memory(),
            sr=self.sr.pin_memory(),
            lengths=self.lengths.pin_memory(),
            num_targets=self.num_targets.pin_memory(),
            spk_labels=self.spk_labels.pin_memory(),
            target_present=self.target_present.pin_memory(),
            cues=self.cues.pin_memory(),
        )
