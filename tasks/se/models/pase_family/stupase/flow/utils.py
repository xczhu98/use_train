# Copyright (c) 2025 ASLP-lab. MIT license.
# Source: https://github.com/ASLP-lab/SenSE/blob/main/src/sense/model/utils.py
# Ported into use_train framework for StuPASE reproduction.

from __future__ import annotations

import torch


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def lens_to_mask(t: torch.Tensor, length: int | None = None) -> torch.Tensor:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: torch.Tensor, frac_lengths: torch.Tensor):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)
