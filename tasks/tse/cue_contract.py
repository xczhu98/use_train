"""TSE 条件线索的样本级与批级类型合同。"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import torch


CueSlot = Literal[
    "speaker_waveform",
    "speaker_embedding",
    "spatial",
    "visual",
    "textual",
]
CueRepresentation = Literal[
    "waveform",
    "embedding",
    "static",
    "track",
    "features",
    "raw",
    "phoneme_ids",
]

CUE_SLOTS: tuple[CueSlot, ...] = (
    "speaker_waveform",
    "speaker_embedding",
    "spatial",
    "visual",
    "textual",
)
_ALLOWED_REPRESENTATIONS: dict[CueSlot, frozenset[str]] = {
    "speaker_waveform": frozenset({"waveform"}),
    "speaker_embedding": frozenset({"embedding"}),
    "spatial": frozenset({"static", "track"}),
    "visual": frozenset({"features", "raw"}),
    "textual": frozenset({"phoneme_ids"}),
}
_TEMPORAL_AXIS: dict[tuple[CueSlot, str], int] = {
    ("speaker_waveform", "waveform"): 0,
    ("spatial", "track"): 1,
    ("visual", "features"): 1,
    ("visual", "raw"): 3,
    ("textual", "phoneme_ids"): 0,
}
_VALUE_DTYPE: dict[tuple[CueSlot, str], torch.dtype] = {
    ("speaker_waveform", "waveform"): torch.float32,
    ("speaker_embedding", "embedding"): torch.float32,
    ("spatial", "static"): torch.float32,
    ("spatial", "track"): torch.float32,
    ("visual", "features"): torch.float32,
    ("visual", "raw"): torch.uint8,
    ("textual", "phoneme_ids"): torch.int64,
}
_TIMEBASE_DTYPE: dict[tuple[CueSlot, str], torch.dtype] = {
    ("spatial", "track"): torch.int64,
    ("visual", "features"): torch.float64,
    ("visual", "raw"): torch.float64,
}


def _require_slot(slot: str) -> CueSlot:
    if slot not in CUE_SLOTS:
        raise ValueError(f"unknown TSE cue slot: {slot!r}")
    return slot  # type: ignore[return-value]


def _validate_representation(slot: CueSlot, representation: str) -> None:
    if representation not in _ALLOWED_REPRESENTATIONS[slot]:
        raise ValueError(
            f"{slot} representation must be one of "
            f"{sorted(_ALLOWED_REPRESENTATIONS[slot])}, got {representation!r}"
        )


def _validate_shape(slot: CueSlot, representation: str, shape: tuple[int, ...]) -> None:
    if not isinstance(shape, tuple) or not shape or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape
    ):
        raise ValueError(f"{slot} value_shape must be a non-empty tuple of positive ints")
    valid = {
        ("speaker_waveform", "waveform"): len(shape) == 1,
        ("speaker_embedding", "embedding"): len(shape) == 1,
        ("spatial", "static"): shape == (2,),
        ("spatial", "track"): len(shape) == 2 and shape[0] == 2,
        ("visual", "features"): len(shape) == 2,
        ("visual", "raw"): len(shape) == 4 and shape[2] == 3,
        ("textual", "phoneme_ids"): len(shape) == 1,
    }[(slot, representation)]
    if not valid:
        raise ValueError(
            f"invalid {slot}/{representation} value_shape: {shape}"
        )


def _tensor(value: Any, *, dtype: torch.dtype, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor-like") from exc
    if dtype.is_floating_point:
        if not tensor.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating-point dtype")
    elif tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {tensor.dtype}")
    return tensor.to(dtype=dtype).contiguous()


def _valid_prefix(tensor: torch.Tensor, axis: int, length: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[axis] = slice(0, length)
    return tensor[tuple(slices)]


@dataclass(frozen=True)
class TSECue:
    """一条目标的一个 typed cue；缺失 cue 使用 ``value=None``。"""

    representation: CueRepresentation
    value: Any | None
    present: bool
    length: int | None
    timebase: Any | None

    def validate(self, slot: CueSlot, spec: "TSECueSpec") -> None:
        slot = _require_slot(slot)
        spec.validate(slot)
        if self.representation != spec.representation:
            raise ValueError(
                f"{slot} representation mismatch: expected {spec.representation!r}, "
                f"got {self.representation!r}"
            )
        if type(self.present) is not bool:
            raise TypeError(f"{slot}.present must be bool")

        temporal_axis = spec.temporal_axis
        requires_timebase = spec.timebase_dtype is not None
        if not self.present:
            if self.value is not None:
                raise ValueError(f"missing {slot} cue must use value=None")
            expected_length = 0 if temporal_axis is not None else None
            if self.length != expected_length:
                raise ValueError(
                    f"missing {slot} cue must use length={expected_length!r}"
                )
            if self.timebase is not None:
                raise ValueError(f"missing {slot} cue must use timebase=None")
            return

        if self.value is None:
            raise ValueError(f"present {slot} cue requires a value")
        value = _tensor(self.value, dtype=spec.value_dtype, name=f"{slot}.value")
        if value.ndim != len(spec.value_shape):
            raise ValueError(
                f"{slot}.value must have {len(spec.value_shape)} dimensions, "
                f"got shape {tuple(value.shape)}"
            )
        for axis, (actual, configured) in enumerate(zip(value.shape, spec.value_shape)):
            if axis != temporal_axis and actual != configured:
                raise ValueError(
                    f"{slot}.value shape {tuple(value.shape)} does not match configured "
                    f"non-temporal shape {spec.value_shape}"
                )

        if temporal_axis is None:
            if self.length is not None:
                raise ValueError(f"static {slot} cue must use length=None")
        else:
            if isinstance(self.length, bool) or not isinstance(self.length, int):
                raise TypeError(f"{slot}.length must be an int")
            if self.length < 1 or self.length > value.shape[temporal_axis]:
                raise ValueError(
                    f"{slot}.length must be in [1, {value.shape[temporal_axis]}]"
                )

        valid_length = self.length if temporal_axis is not None else None
        valid_value = (
            _valid_prefix(value, temporal_axis, valid_length)
            if temporal_axis is not None and valid_length is not None
            else value
        )
        if value.dtype.is_floating_point and not bool(torch.isfinite(valid_value).all().item()):
            raise ValueError(f"{slot}.value contains NaN/Inf in its valid region")

        if requires_timebase:
            if self.timebase is None:
                raise ValueError(f"present {slot}/{self.representation} requires timebase")
            timebase = _tensor(
                self.timebase,
                dtype=spec.timebase_dtype,
                name=f"{slot}.timebase",
            )
            if timebase.ndim != 1 or timebase.shape[0] != value.shape[temporal_axis]:
                raise ValueError(
                    f"{slot}.timebase must have shape [{value.shape[temporal_axis]}]"
                )
            assert valid_length is not None
            if timebase.dtype.is_floating_point and not bool(
                torch.isfinite(timebase[:valid_length]).all().item()
            ):
                raise ValueError(f"{slot}.timebase contains NaN/Inf in its valid region")
            if bool((timebase[:valid_length] < 0).any().item()):
                raise ValueError(f"{slot}.timebase must be relative to mixture start and non-negative")
            if valid_length > 1 and bool(
                (timebase[1:valid_length] < timebase[: valid_length - 1]).any().item()
            ):
                raise ValueError(f"{slot}.timebase must be monotonic non-decreasing")
        elif self.timebase is not None:
            raise ValueError(f"{slot}/{self.representation} must use timebase=None")


@dataclass(frozen=True)
class TSECueSpec:
    """一个启用 cue 的表示和批内物化容量。"""

    representation: CueRepresentation
    value_shape: tuple[int, ...]

    def validate(self, slot: CueSlot) -> None:
        slot = _require_slot(slot)
        _validate_representation(slot, self.representation)
        _validate_shape(slot, self.representation, self.value_shape)

    @property
    def temporal_axis(self) -> int | None:
        return _TEMPORAL_AXIS.get((self._slot(), self.representation))

    @property
    def value_dtype(self) -> torch.dtype:
        return _VALUE_DTYPE[(self._slot(), self.representation)]

    @property
    def timebase_dtype(self) -> torch.dtype | None:
        return _TIMEBASE_DTYPE.get((self._slot(), self.representation))

    def _slot(self) -> CueSlot:
        matches = [
            slot for slot in CUE_SLOTS if self.representation in _ALLOWED_REPRESENTATIONS[slot]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"representation {self.representation!r} does not identify one cue slot"
            )
        return matches[0]


@dataclass(frozen=True)
class TSECues:
    """样本级固定 cue slots；``None`` 表示该 slot 未启用。"""

    speaker_waveform: TSECue | None = None
    speaker_embedding: TSECue | None = None
    spatial: TSECue | None = None
    visual: TSECue | None = None
    textual: TSECue | None = None

    def validate(self, specs: "TSECueSpecs") -> None:
        specs.validate()
        for slot in CUE_SLOTS:
            cue = getattr(self, slot)
            spec = getattr(specs, slot)
            if spec is None:
                if cue is not None:
                    raise ValueError(f"{slot} cue is provided but the slot is not enabled")
            elif cue is None:
                raise ValueError(
                    f"enabled {slot} slot must use TSECue, including when cue is missing"
                )
            else:
                cue.validate(slot, spec)


@dataclass(frozen=True)
class TSECueSpecs:
    """collate 的固定 cue shape 配置。"""

    speaker_waveform: TSECueSpec | None = None
    speaker_embedding: TSECueSpec | None = None
    spatial: TSECueSpec | None = None
    visual: TSECueSpec | None = None
    textual: TSECueSpec | None = None

    def validate(self) -> None:
        for slot in CUE_SLOTS:
            spec = getattr(self, slot)
            if spec is not None:
                if not isinstance(spec, TSECueSpec):
                    raise TypeError(f"{slot} spec must be TSECueSpec or None")
                spec.validate(slot)


@dataclass
class TSECueBatch:
    """按目标行对齐的一个 cue batch。"""

    representation: CueRepresentation
    value: torch.Tensor
    present: torch.Tensor
    lengths: torch.Tensor | None
    timebase: torch.Tensor | None

    def validate(self, slot: CueSlot, spec: TSECueSpec, rows: int) -> None:
        spec.validate(slot)
        if self.representation != spec.representation:
            raise ValueError(f"{slot} batch representation does not match its spec")
        if not isinstance(self.value, torch.Tensor):
            raise TypeError(f"{slot}.value must be a tensor")
        if self.value.ndim != len(spec.value_shape) + 1 or self.value.shape[0] != rows:
            raise ValueError(
                f"{slot}.value must have row axis [{rows}] and "
                f"{len(spec.value_shape)} value dimensions, got {tuple(self.value.shape)}"
            )
        temporal_axis = spec.temporal_axis
        for axis, configured in enumerate(spec.value_shape):
            actual = int(self.value.shape[axis + 1])
            if axis == temporal_axis:
                if actual < configured:
                    raise ValueError(
                        f"{slot}.value temporal capacity {actual} is smaller than "
                        f"configured fallback {configured}"
                    )
            elif actual != configured:
                raise ValueError(
                    f"{slot}.value non-temporal shape must match {spec.value_shape}, "
                    f"got {tuple(self.value.shape[1:])}"
                )
        if self.value.dtype != spec.value_dtype:
            raise TypeError(f"{slot}.value must use {spec.value_dtype}")
        if not isinstance(self.present, torch.Tensor) or (
            tuple(self.present.shape) != (rows,) or self.present.dtype != torch.bool
        ):
            raise TypeError(f"{slot}.present must be torch.bool with shape [{rows}]")

        capacity = (
            int(self.value.shape[temporal_axis + 1])
            if temporal_axis is not None
            else None
        )
        if temporal_axis is None:
            if self.lengths is not None:
                raise ValueError(f"static {slot} batch must use lengths=None")
        else:
            if not isinstance(self.lengths, torch.Tensor) or tuple(self.lengths.shape) != (rows,):
                raise ValueError(f"{slot}.lengths must have shape [{rows}]")
            if self.lengths.dtype != torch.int32:
                raise TypeError(f"{slot}.lengths must use torch.int32")

        if spec.timebase_dtype is None:
            if self.timebase is not None:
                raise ValueError(f"{slot}/{self.representation} batch must use timebase=None")
        else:
            assert capacity is not None
            if not isinstance(self.timebase, torch.Tensor) or tuple(self.timebase.shape) != (
                rows,
                capacity,
            ):
                raise ValueError(f"{slot}.timebase must have shape [{rows}, {capacity}]")
            if self.timebase.dtype != spec.timebase_dtype:
                raise TypeError(f"{slot}.timebase must use {spec.timebase_dtype}")

        tensors = [self.value, self.present]
        if self.lengths is not None:
            tensors.append(self.lengths)
        if self.timebase is not None:
            tensors.append(self.timebase)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError(f"{slot} batch tensors must be on one device")
        checks: list[tuple[torch.Tensor, str]] = []
        if temporal_axis is not None:
            assert self.lengths is not None and capacity is not None
            invalid = (self.present & ((self.lengths < 1) | (self.lengths > capacity))) | (
                ~self.present & (self.lengths != 0)
            )
            checks.append((invalid.any(), f"{slot}.lengths must be positive for present rows and zero otherwise"))
        if self.value.dtype.is_floating_point:
            checks.append((~torch.isfinite(self.value).all(), f"{slot}.value must contain only finite values"))
        if self.timebase is not None and self.timebase.dtype.is_floating_point:
            checks.append((~torch.isfinite(self.timebase).all(), f"{slot}.timebase must contain only finite values"))
        nonzero = self.value != 0
        row_nonzero = nonzero.flatten(1).any(dim=1)
        checks.append(((row_nonzero & ~self.present).any(), f"missing {slot} rows must use zero value placeholders"))
        if self.timebase is not None:
            checks.append((((self.timebase != 0).any(dim=1) & ~self.present).any(), f"missing {slot} rows must use zero timebase placeholders"))
        if temporal_axis is not None:
            valid = torch.arange(capacity, device=self.value.device)[None, :] < self.lengths[:, None]
            temporal_nonzero = nonzero.movedim(temporal_axis + 1, 1).flatten(2).any(dim=2) if nonzero.ndim > 2 else nonzero
            checks.append(((temporal_nonzero & ~valid & self.present[:, None]).any(), f"{slot}.value samples after valid length must be zero"))
            if self.timebase is not None:
                checks.extend((
                    (((self.timebase < 0) & valid).any(), f"{slot}.timebase must be relative to mixture start and non-negative"),
                    (((self.timebase[:, 1:] < self.timebase[:, :-1]) & valid[:, 1:]).any(), f"{slot}.timebase must be monotonic non-decreasing"),
                    (((self.timebase != 0) & ~valid & self.present[:, None]).any(), f"{slot}.timebase samples after valid length must be zero"),
                ))
        failures = torch.stack([check for check, _ in checks]).detach().cpu().tolist()
        for failed, (_, message) in zip(failures, checks):
            if failed:
                raise ValueError(message)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TSECueBatch":
        return replace(
            self,
            value=self.value.to(device=device, non_blocking=non_blocking),
            present=self.present.to(device=device, non_blocking=non_blocking),
            lengths=(
                self.lengths.to(device=device, non_blocking=non_blocking)
                if self.lengths is not None
                else None
            ),
            timebase=(
                self.timebase.to(device=device, non_blocking=non_blocking)
                if self.timebase is not None
                else None
            ),
        )

    def pin_memory(self) -> "TSECueBatch":
        return replace(
            self,
            value=self.value.pin_memory(),
            present=self.present.pin_memory(),
            lengths=self.lengths.pin_memory() if self.lengths is not None else None,
            timebase=self.timebase.pin_memory() if self.timebase is not None else None,
        )


@dataclass
class TSECueBatches:
    """批级固定 cue slots。"""

    speaker_waveform: TSECueBatch | None = None
    speaker_embedding: TSECueBatch | None = None
    spatial: TSECueBatch | None = None
    visual: TSECueBatch | None = None
    textual: TSECueBatch | None = None

    def validate(self, specs: TSECueSpecs, rows: int) -> None:
        specs.validate()
        for slot in CUE_SLOTS:
            cue_batch = getattr(self, slot)
            spec = getattr(specs, slot)
            if spec is None:
                if cue_batch is not None:
                    raise ValueError(f"{slot} batch exists but the slot is not enabled")
            elif cue_batch is None:
                raise ValueError(f"enabled {slot} slot is missing from cue batches")
            else:
                cue_batch.validate(slot, spec, rows)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TSECueBatches":
        values = {
            field.name: (
                getattr(self, field.name).to(device, non_blocking=non_blocking)
                if getattr(self, field.name) is not None
                else None
            )
            for field in fields(self)
        }
        return replace(self, **values)

    def pin_memory(self) -> "TSECueBatches":
        values = {
            field.name: (
                getattr(self, field.name).pin_memory()
                if getattr(self, field.name) is not None
                else None
            )
            for field in fields(self)
        }
        return replace(self, **values)


def collate_tse_cues(rows: list[TSECues], specs: TSECueSpecs) -> TSECueBatches:
    """按动态时间容量物化 cue；输入与输出合同各校验一次。"""
    result = _materialize_tse_cues(rows, specs)
    result.validate(specs, len(rows))
    return result


def _materialize_tse_cues(rows: list[TSECues], specs: TSECueSpecs) -> TSECueBatches:
    """校验样本后物化；调用边界负责校验完整批合同。"""
    specs.validate()
    for cues in rows:
        if not isinstance(cues, TSECues):
            raise TypeError("target cues must be TSECues")
        cues.validate(specs)

    batches: dict[str, TSECueBatch | None] = {}
    row_count = len(rows)
    for slot in CUE_SLOTS:
        spec = getattr(specs, slot)
        if spec is None:
            batches[slot] = None
            continue
        temporal_axis = spec.temporal_axis
        batch_shape = list(spec.value_shape)
        if temporal_axis is not None:
            batch_shape[temporal_axis] = max(
                [
                    batch_shape[temporal_axis],
                    *(
                        int(torch.as_tensor(getattr(cues, slot).value).shape[temporal_axis])
                        for cues in rows
                        if getattr(cues, slot).present
                    ),
                ]
            )
        value = torch.zeros((row_count, *batch_shape), dtype=spec.value_dtype)
        present = torch.zeros(row_count, dtype=torch.bool)
        lengths = (
            torch.zeros(row_count, dtype=torch.int32) if temporal_axis is not None else None
        )
        capacity = batch_shape[temporal_axis] if temporal_axis is not None else None
        timebase = (
            torch.zeros((row_count, capacity), dtype=spec.timebase_dtype)
            if spec.timebase_dtype is not None and capacity is not None
            else None
        )

        for row, cues in enumerate(rows):
            cue = getattr(cues, slot)
            assert cue is not None
            if not cue.present:
                continue
            source = _tensor(cue.value, dtype=spec.value_dtype, name=f"{slot}.value")
            present[row] = True
            if temporal_axis is None:
                value[row].copy_(source)
                continue
            assert cue.length is not None and lengths is not None
            lengths[row] = cue.length
            destination = value[row]
            source_valid = _valid_prefix(source, temporal_axis, cue.length)
            slices = [slice(None)] * destination.ndim
            slices[temporal_axis] = slice(0, cue.length)
            destination[tuple(slices)].copy_(source_valid)
            if timebase is not None:
                cue_timebase = _tensor(
                    cue.timebase,
                    dtype=spec.timebase_dtype,
                    name=f"{slot}.timebase",
                )
                timebase[row, : cue.length].copy_(cue_timebase[: cue.length])

        batch = TSECueBatch(
            representation=spec.representation,
            value=value,
            present=present,
            lengths=lengths,
            timebase=timebase,
        )
        batches[slot] = batch

    return TSECueBatches(**batches)
