"""Strict collate contract for Speech Separation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from core.data.source_separation_batch import SSBatch


def _to_channels(value: torch.Tensor, *, location: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{location} must be a tensor")
    if value.ndim == 1:
        return value.float().unsqueeze(0)
    if value.ndim == 2 and value.shape[0] >= 1:
        return value.float()
    raise ValueError(f"{location} must have shape [T] or [C, T]")


def ss_collate_fn(batch: Sequence[dict]) -> SSBatch:
    if not batch:
        raise ValueError("SS batch must not be empty")
    source_count = len(batch[0]["sources"])
    if source_count < 1 or any(len(item["sources"]) != source_count for item in batch):
        raise ValueError("SS collate requires a fixed source count per batch")
    for index, item in enumerate(batch):
        for name in ("sr", "length"):
            value = item[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"batch[{index}].{name} must be an exact positive int")
    sample_rates = [item["sr"] for item in batch]
    if len(set(sample_rates)) != 1:
        raise ValueError(f"SS sample rates must match within a batch: {sample_rates}")
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.int64)
    max_length = int(lengths.max().item())
    mixtures = []
    sources = []
    channel_count = None
    for index, item in enumerate(batch):
        mixture = _to_channels(item["mixture"], location=f"batch[{index}].mixture")
        if channel_count is None:
            channel_count = mixture.shape[0]
        if mixture.shape[0] != channel_count:
            raise ValueError("SS collate requires a fixed microphone count per batch")
        source_rows = [
            _to_channels(source, location=f"batch[{index}].sources[{source_index}]")
            for source_index, source in enumerate(item["sources"])
        ]
        if any(source.shape[0] != 1 for source in source_rows):
            raise ValueError("SS targets must remain reference single-channel waveforms")
        if int(lengths[index]) > mixture.shape[-1] or any(
            int(lengths[index]) > source.shape[-1] for source in source_rows
        ):
            raise ValueError(f"batch[{index}] length exceeds a waveform")
        valid_length = int(lengths[index])
        mixtures.append(F.pad(mixture[:, :valid_length], (0, max_length - valid_length)))
        sources.append(
            torch.stack(
                [F.pad(source[0, :valid_length], (0, max_length - valid_length)) for source in source_rows]
            )
        )
    mixture_batch = torch.stack(mixtures)
    sources_batch = torch.stack(sources)
    if channel_count == 1:
        mixture_batch = mixture_batch[:, 0]
    source_ids = None
    if any("source_ids" in item for item in batch):
        if not all("source_ids" in item for item in batch):
            raise ValueError(
                "SS source_ids must be present for every item or omitted for every item"
            )
        for index, item in enumerate(batch):
            values = item["source_ids"]
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"batch[{index}].source_ids must contain non-empty strings")
        source_ids = tuple(tuple(item["source_ids"]) for item in batch)
    result = SSBatch(
        mixture=mixture_batch,
        sources=sources_batch,
        lengths=lengths,
        source_mask=torch.ones(len(batch), source_count, dtype=torch.bool),
        sr=torch.tensor(sample_rates, dtype=torch.int64),
        source_ids=source_ids,
    )
    result.validate()
    return result
