"""波形 Batch 的形状与元数据校验；波形有限值策略由 collate/Trainer 管理。"""

from __future__ import annotations

import torch


_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def validate_waveform(
    value: torch.Tensor, *, name: str, spatial: bool = False
) -> tuple[int, int]:
    """检查浮点波形布局并返回主样本数、采样点数。"""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    allowed_dims = (2, 3) if spatial else (2,)
    if value.ndim not in allowed_dims or any(size < 1 for size in value.shape):
        layout = "[B,T] or [B,C,T]" if spatial else "[B,T]"
        raise ValueError(f"{name} must have non-empty shape {layout}, got {tuple(value.shape)}")
    return int(value.shape[0]), int(value.shape[-1])


def validate_waveform_lengths(
    lengths: torch.Tensor,
    *,
    batch_size: int,
    max_length: int,
    context: str,
) -> None:
    """校验 waveform batch 的逐样本有效长度元数据。"""
    if not isinstance(lengths, torch.Tensor):
        raise TypeError(f"{context} lengths must be a torch.Tensor")
    if lengths.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{context} lengths must use an integer dtype, got {lengths.dtype}")
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError(
            f"{context} lengths must have shape ({batch_size},), got {tuple(lengths.shape)}"
        )
    if bool(((lengths < 1) | (lengths > max_length)).any().item()):
        raise ValueError(f"{context} lengths must be in [1, {max_length}]")


def validate_sample_rates(
    value: torch.Tensor, *, batch_size: int, name: str
) -> None:
    """接受正整数标量或统一的逐样本采样率。"""
    if not isinstance(value, torch.Tensor) or value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must be an integer tensor")
    if value.ndim != 0 and value.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape [{batch_size}]")
    rates = value.reshape(-1)
    if bool((rates <= 0).any()) or not bool((rates == rates[0]).all()):
        raise ValueError(f"{name} must contain one uniform positive sample rate")


def validate_batch_flag(
    value: torch.Tensor | None, *, batch_size: int, name: str
) -> None:
    """可选门控标志存在时必须是逐样本 bool。"""
    if value is None:
        return
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a bool tensor")
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape [{batch_size}]")
