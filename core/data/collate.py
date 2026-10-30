"""Collate function: pad variable-length audio to max length within batch.

Dataset 约定（offline / dynamic）：
  __getitem__ → (clean, noisy, sr, length) 4 元组，或带三分支门控 flag 的 7 元组
  ``(..., is_pure_noise, is_pure_echo, is_pure_echo_noise)``（bool / 0/1）。
  - clean/target: torch 或 ndarray，仅接受 mono ``(T)``/``(1,T)``/``(T,1)``；collate 后为 ``(B,T_max)``。
  - noisy: 默认仅接受 mono 布局；显式 ``allow_spatial_noisy=True`` 时还接受 channel-first ``(C,T)``，collate 后为 ``(B,C,T_max)``。
  - sr: int 或与 batch 一致的标量；同 batch 内必须相同（GroupedBatchSampler 按 SR 分桶）。
  - length: 有效采样点数，≤ 该条波形时间长度；输出在 length 后清零。
  - 三分支门控：仅 7 元组携带；输入必须为标量，collate 后各为 ``(B,)`` bool。
  - 任一条样本为 ``None`` 时立即报错，不静默改变 batch size。

Output: 已校验的 SEBatch；三分支门控可省略。空间 noisy 只能由调用方显式 opt-in。
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import torch
import torch.nn.functional as F

from core.data.speech_enhancement_batch import SEBatch


def _to_1d_wav(x, name: str, idx: int) -> torch.Tensor:
    """将明确的 mono 布局 ``(T)``、``(1,T)`` 或 ``(T,1)`` 规范化为 1-D。"""
    t = torch.as_tensor(x).float()
    shape = tuple(t.shape)
    if t.dim() == 1 and t.shape[0] > 0:
        return t.contiguous()
    if t.dim() == 2 and t.numel() > 0:
        if t.shape[0] == 1:
            return t[0].contiguous()
        if t.shape[1] == 1:
            return t[:, 0].contiguous()
    raise RuntimeError(
        f"batch[{idx}] {name} must be a non-empty mono waveform with shape "
        f"(T), (1,T), or (T,1), got {shape}"
    )


def _to_spatial_noisy(
    x,
    name: str,
    idx: int,
    *,
    declared_length: int | None = None,
    keep_single_channel: bool = True,
) -> torch.Tensor:
    """将显式空间 waveform 规范化为 channel-first ``(C,T)``。

    ``(C, 1)`` 在空间路径中始终按 ``C`` 个通道保留。任务 collate 可传入
    ``declared_length``，仅在长度大于 1 时把历史 ``(T, 1)`` mono 形式还原为
    1-D；这使 ``C=2, T=1`` 与 legacy mono 兼容路径不再发生形状歧义。
    """
    t = torch.as_tensor(x).float()
    shape = tuple(t.shape)
    if t.dim() == 1 and t.shape[0] > 0:
        return t.unsqueeze(0).contiguous() if keep_single_channel else t.contiguous()
    if t.dim() == 2 and t.numel() > 0:
        if t.shape[0] == 1:
            return t.contiguous() if keep_single_channel else t[0].contiguous()
        if t.shape[1] == 1 and declared_length is not None and declared_length > 1:
            return t[:, 0].contiguous()
        if t.shape[1] > 0:
            return t.contiguous()
    raise RuntimeError(
        f"batch[{idx}] {name} must be a non-empty channel-first waveform with shape "
        f"(T), (1,T), or (C,T), got {shape}"
    )


def collate_fn(
    batch: Sequence[tuple],
    *,
    allow_spatial_noisy: bool = False,
) -> SEBatch:
    """Pad variable-length samples to uniform length and form batch.

    Args:
        batch: list of 4- or 7-tuples from Dataset.__getitem__ (see module docstring).
        allow_spatial_noisy: explicitly retain channel-first noisy waveforms as
            ``(B,C,T_max)``; clean remains ``(B,T_max)``.

    Returns:
        已校验的 SEBatch，包含可选的三个 ``(B,)`` bool 门控。
    """
    if len(batch) == 0:
        raise RuntimeError("collate_fn: batch must not be empty")
    for i, sample in enumerate(batch):
        if sample is None:
            raise RuntimeError(
                f"collate_fn: batch[{i}] is None (load/__getitem__ failure); "
                "SE batches fail fast instead of silently changing batch size"
            )
    for i, s in enumerate(batch):
        if not isinstance(s, tuple) or len(s) < 4:
            raise TypeError(
                f"batch[{i}] must be tuple(clean,noisy,sr,length[, flags...]), got: {type(s)}"
            )
        if len(s) not in (4, 7):
            raise TypeError(
                f"batch[{i}] tuple length must be 4 or 7, got len={len(s)}; "
                "the legacy 5-tuple (is_pure_noise only) is no longer accepted"
            )
    n_fields = len(batch[0])
    if not all(len(s) == n_fields for s in batch):
        raise RuntimeError(
            "collate_fn: mixed tuple lengths in one batch (must be all 4 or all 7)."
        )

    if type(allow_spatial_noisy) is not bool:
        raise TypeError("allow_spatial_noisy must be bool")
    cleans = [_to_1d_wav(s[0], "clean", i) for i, s in enumerate(batch)]
    noisy_parser = _to_spatial_noisy if allow_spatial_noisy else _to_1d_wav
    noisys = [noisy_parser(s[1], "noisy", i) for i, s in enumerate(batch)]
    if allow_spatial_noisy:
        channels = {int(noisy.shape[0]) for noisy in noisys}
        if len(channels) != 1:
            raise RuntimeError(
                f"Spatial noisy channel count must match within a batch, got {sorted(channels)}"
            )

    def _one_integer_scalar(v: object, name: str, idx: int) -> int:
        """接受整数标量，禁止 bool、浮点和隐式截断。"""
        if isinstance(v, Integral) and not isinstance(v, bool):
            return int(v)
        if isinstance(v, torch.Tensor):
            is_integer_dtype = (
                v.dtype is not torch.bool
                and not v.dtype.is_floating_point
                and not v.dtype.is_complex
            )
            if v.numel() == 1 and is_integer_dtype:
                return int(v.item())
        raise TypeError(
            f"batch[{idx}] {name} must be a non-bool integer scalar or a one-element "
            f"integer-dtype tensor, got {v!r}"
        )

    srs = [_one_integer_scalar(s[2], "sr", i) for i, s in enumerate(batch)]
    lens = [_one_integer_scalar(s[3], "length", i) for i, s in enumerate(batch)]

    def _one_bool_flag(v: object, name: str, idx: int) -> bool:
        try:
            flag = torch.as_tensor(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"batch[{idx}] {name} flag must be a strict bool or finite numeric 0/1"
            ) from exc
        if flag.dim() != 0:
            raise TypeError(
                f"batch[{idx}] {name} flag must be scalar, got shape {tuple(flag.shape)}"
            )
        value = flag.item()
        if isinstance(value, bool):
            return value
        if (
            not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value not in (0, 1)
        ):
            raise ValueError(
                f"batch[{idx}] {name} flag must be a strict bool or finite numeric 0/1"
            )
        return bool(value)

    if n_fields == 7:
        pn_list = [
            _one_bool_flag(s[4], "is_pure_noise", i) for i, s in enumerate(batch)
        ]
        pe_list = [
            _one_bool_flag(s[5], "is_pure_echo", i) for i, s in enumerate(batch)
        ]
        pen_list = [
            _one_bool_flag(s[6], "is_pure_echo_noise", i)
            for i, s in enumerate(batch)
        ]

    if not all(sr == srs[0] for sr in srs):
        raise RuntimeError(f"Sample rates must match within a batch, got {set(srs)}")
    if any(sr <= 0 for sr in srs):
        raise RuntimeError(f"Invalid sample rate in batch: {srs}")
    if any(l <= 0 for l in lens):
        raise RuntimeError(f"Invalid lengths in batch: {lens}")

    for i, (c, n, le) in enumerate(zip(cleans, noisys, lens)):
        if c.shape[-1] != n.shape[-1]:
            raise RuntimeError(
                f"Sample {i}: clean/noisy length mismatch clean={tuple(c.shape)} noisy={tuple(n.shape)}"
            )
        if le > c.shape[-1]:
            raise RuntimeError(
                f"Sample {i}: length={le} exceeds waveform len={c.shape[-1]} (Dataset 约定 length≤T)"
            )
        if not torch.isfinite(c).all() or not torch.isfinite(n).all():
            raise RuntimeError(
                f"Sample {i} contains NaN/Inf (strict_load=True 时应在 Dataset 已拦截；检查自定义 Dataset)"
            )

    # 仅由已验证的有效长度决定 batch 上界，避免损坏样本的物理尾部撑大内存。
    mx = max(lens)

    def _pad_stack(lst: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [F.pad(x[..., :length], (0, mx - length)) for x, length in zip(lst, lens)]
        )

    out = SEBatch(
        clean=_pad_stack(cleans),
        noisy=_pad_stack(noisys),
        sr=torch.tensor(srs[0], dtype=torch.int32),
        lengths=torch.tensor(lens, dtype=torch.int32),
        is_pure_noise=torch.tensor(pn_list, dtype=torch.bool) if n_fields == 7 else None,
        is_pure_echo=torch.tensor(pe_list, dtype=torch.bool) if n_fields == 7 else None,
        is_pure_echo_noise=torch.tensor(pen_list, dtype=torch.bool) if n_fields == 7 else None,
    )
    out.validate()
    return out
