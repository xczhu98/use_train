"""Canonical dict-item to validated ``TSEBatch`` collation."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from core.data.collate import _to_1d_wav, _to_spatial_noisy
from tasks.tse.cue_contract import (
    TSECueSpecs,
    TSECues,
    _materialize_tse_cues,
)
from tasks.tse.data.batch import TSEBatch

_SAMPLE_KEYS = frozenset(
    {"key", "mixture", "sr", "length", "targets", "geometry", "timebase", "provenance"}
)
_TARGET_KEYS = frozenset(
    {"target", "speaker_id", "spk_label", "target_present", "cues"}
)


def _meta_int(value: Any, name: str, index: int) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                f"batch[{index}] meta field {name!r} must be a scalar integer tensor"
            )
        return int(value.item())
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"batch[{index}] meta field {name!r} is not an int: {value!r}")
    return value


def make_tse_target(
    *,
    target: Any,
    cues: TSECues,
    speaker_id: str | None,
    spk_label: int = -1,
    target_present: bool = True,
) -> dict[str, Any]:
    """构造 canonical target dict。"""
    return {
        "target": target,
        "speaker_id": speaker_id,
        "spk_label": spk_label,
        "target_present": target_present,
        "cues": cues,
    }


def make_tse_sample(
    *,
    key: str,
    mixture: Any,
    sr: int,
    length: int,
    targets: Sequence[dict[str, Any]],
    geometry: Any | None = None,
    timebase: Any | None = None,
    provenance: Any | None = None,
) -> dict[str, Any]:
    """构造 canonical dataset item；完整校验在 ``tse_collate_fn`` 执行。"""
    return {
        "key": key,
        "mixture": mixture,
        "sr": sr,
        "length": length,
        "targets": list(targets),
        "geometry": geometry,
        "timebase": timebase,
        "provenance": provenance,
    }


def _normalize_sample(
    sample: Any,
    index: int,
) -> tuple[
    str,
    torch.Tensor,
    int,
    int,
    list[tuple[torch.Tensor, str | None, int, bool, TSECues]],
    Any | None,
    Any | None,
    Any | None,
]:
    if not isinstance(sample, dict):
        raise TypeError(
            f"batch[{index}] must be a canonical dict item; tuple TSE samples are not accepted"
        )
    missing = {"key", "mixture", "sr", "length", "targets"} - set(sample)
    if missing:
        raise TypeError(f"batch[{index}] dict sample missing keys: {sorted(missing)}")
    unknown = set(sample) - _SAMPLE_KEYS
    if unknown:
        raise TypeError(f"batch[{index}] dict sample has unknown keys: {sorted(unknown)}")
    key = sample["key"]
    if not isinstance(key, str) or not key:
        raise TypeError(f"batch[{index}] key must be a non-empty string")

    sr = _meta_int(sample["sr"], "sr", index)
    length = _meta_int(sample["length"], "length", index)
    mixture = _to_spatial_noisy(
        sample["mixture"],
        "mixture",
        index,
        declared_length=length,
        keep_single_channel=True,
    )
    if mixture.ndim == 1:
        mixture = mixture.unsqueeze(0)
    raw_targets = sample["targets"]
    if not isinstance(raw_targets, (list, tuple)) or not raw_targets:
        raise TypeError(f"batch[{index}] 'targets' must be a non-empty list")

    targets: list[tuple[torch.Tensor, str | None, int, bool, TSECues]] = []
    for target_index, raw_target in enumerate(raw_targets):
        location = f"batch[{index}] targets[{target_index}]"
        if not isinstance(raw_target, dict):
            raise TypeError(f"{location} must be a canonical target dict")
        missing_target = _TARGET_KEYS - set(raw_target)
        if missing_target:
            raise TypeError(f"{location} missing keys: {sorted(missing_target)}")
        unknown_target = set(raw_target) - _TARGET_KEYS
        if unknown_target:
            raise TypeError(f"{location} has unknown keys: {sorted(unknown_target)}")
        speaker_id = raw_target["speaker_id"]
        if speaker_id is not None and (
            not isinstance(speaker_id, str) or not speaker_id
        ):
            raise TypeError(f"{location} speaker_id must be a non-empty string or None")
        target_present = raw_target["target_present"]
        if type(target_present) is not bool:
            raise TypeError(f"{location} target_present must be bool")
        cues = raw_target["cues"]
        if not isinstance(cues, TSECues):
            raise TypeError(f"{location} cues must be TSECues")
        targets.append(
            (
                _to_1d_wav(raw_target["target"], f"targets[{target_index}].target", index),
                speaker_id,
                _meta_int(raw_target["spk_label"], f"targets[{target_index}].spk_label", index),
                target_present,
                cues,
            )
        )
    return (
        key,
        mixture,
        sr,
        length,
        targets,
        sample.get("geometry"),
        sample.get("timebase"),
        sample.get("provenance"),
    )


def _pad_stack(
    values: list[torch.Tensor],
    maximum: int,
    valid_lengths: list[int],
) -> torch.Tensor:
    rows = []
    for value, valid_length in zip(values, valid_lengths):
        value = value.clone()
        value[..., valid_length:] = 0
        rows.append(F.pad(value, (0, maximum - value.shape[-1])))
    return torch.stack(rows)


def tse_collate_fn(
    batch: Sequence[dict[str, Any]],
    *,
    cue_specs: TSECueSpecs | None = None,
    placeholder_sr: int = 16000,
    placeholder_samples: int = 16000,
    raise_on_all_failed: bool = False,
) -> TSEBatch:
    """将唯一 canonical dict item 链路打包为 validated ``TSEBatch``。

    ``placeholder_*`` 与 ``raise_on_all_failed`` 仅保留 DataLoader 绑定签名，
    不生成占位或改变 None 的 fail-fast 语义。
    """
    del placeholder_sr, placeholder_samples, raise_on_all_failed
    if not batch:
        raise RuntimeError("tse_collate_fn: empty batch (no samples to collate)")
    none_count = sum(sample is None for sample in batch)
    if none_count == len(batch):
        raise RuntimeError(
            f"tse_collate_fn: all {len(batch)} samples were None (load/__getitem__ failure)"
        )
    if none_count:
        raise RuntimeError(
            f"tse_collate_fn: {none_count}/{len(batch)} samples were None; refusing to "
            "shrink a training batch because rank-local filtering breaks DDP sample weighting"
        )
    cue_specs = cue_specs if cue_specs is not None else TSECueSpecs()
    if not isinstance(cue_specs, TSECueSpecs):
        raise TypeError("cue_specs must be TSECueSpecs")
    cue_specs.validate()

    mixtures: list[torch.Tensor] = []
    sample_rates: list[int] = []
    lengths: list[int] = []
    num_targets: list[int] = []
    targets: list[torch.Tensor] = []
    spk_labels: list[int] = []
    target_present: list[bool] = []
    cue_rows: list[TSECues] = []
    keys: list[str] = []
    speaker_ids: list[str | None] = []
    geometry: list[Any | None] = []
    timebase: list[Any | None] = []
    provenance: list[Any | None] = []

    for index, sample in enumerate(batch):
        (
            key,
            mixture,
            sr,
            length,
            sample_targets,
            sample_geometry,
            sample_timebase,
            sample_provenance,
        ) = _normalize_sample(sample, index)
        if sr <= 0:
            raise RuntimeError(f"batch[{index}]: sample rate must be > 0, got {sr}")
        if length <= 0:
            raise RuntimeError(f"batch[{index}]: length must be > 0, got {length}")
        if length > int(mixture.shape[-1]):
            raise RuntimeError(
                f"batch[{index}]: length={length} exceeds waveform len={mixture.shape[-1]}"
            )
        for target_index, (target, _, _, is_present, cues) in enumerate(sample_targets):
            if target.shape[-1] != mixture.shape[-1]:
                raise RuntimeError(
                    f"batch[{index}] targets[{target_index}] mixture/target length mismatch: "
                    f"mixture T={mixture.shape[-1]}, target T={target.shape[-1]}"
                )
            if not is_present and bool((target[:length] != 0).any().item()):
                raise RuntimeError(
                    f"batch[{index}] targets[{target_index}] target_present=False "
                    "requires zero target supervision"
                )

        mixture = mixture.clone()
        mixture[..., length:] = 0
        if not bool(torch.isfinite(mixture).all().item()):
            raise RuntimeError(f"batch[{index}] mixture contains NaN/Inf")
        for target_index, (
            target,
            speaker_id,
            spk_label,
            is_present,
            cues,
        ) in enumerate(sample_targets):
            target = target.clone()
            target[length:] = 0
            if not bool(torch.isfinite(target).all().item()):
                raise RuntimeError(
                    f"batch[{index}] targets[{target_index}] target contains NaN/Inf"
                )
            targets.append(target)
            speaker_ids.append(speaker_id)
            spk_labels.append(spk_label)
            target_present.append(is_present)
            cue_rows.append(cues)

        mixtures.append(mixture)
        sample_rates.append(sr)
        lengths.append(length)
        num_targets.append(len(sample_targets))
        keys.append(key)
        geometry.append(sample_geometry)
        timebase.append(sample_timebase)
        provenance.append(sample_provenance)

    if not all(sr == sample_rates[0] for sr in sample_rates):
        raise RuntimeError(
            f"Sample rates must match within a batch, got {set(sample_rates)}"
        )
    channel_counts = {int(mixture.shape[0]) for mixture in mixtures}
    if len(channel_counts) != 1:
        raise RuntimeError(
            f"TSE observation channel counts must match within a batch, got {channel_counts}"
        )

    maximum_time = max(int(mixture.shape[-1]) for mixture in mixtures)
    target_lengths = [
        length for length, count in zip(lengths, num_targets) for _ in range(count)
    ]
    result = TSEBatch(
        mixture=_pad_stack(mixtures, maximum_time, lengths),
        target=_pad_stack(targets, maximum_time, target_lengths),
        sr=torch.tensor(sample_rates[0], dtype=torch.int32),
        lengths=torch.tensor(lengths, dtype=torch.int32),
        num_targets=torch.tensor(num_targets, dtype=torch.int32),
        spk_labels=torch.tensor(spk_labels, dtype=torch.int64),
        target_present=torch.tensor(target_present, dtype=torch.bool),
        cues=_materialize_tse_cues(cue_rows, cue_specs),
        cue_specs=cue_specs,
        keys=tuple(keys),
        speaker_ids=tuple(speaker_ids),
        geometry=tuple(geometry),
        timebase=tuple(timebase),
        provenance=tuple(provenance),
    )
    result.validate()
    return result
