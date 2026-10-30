"""AEC 文件推理的部署设备策略与 runtime 迁移。

根入口的批量路径与 canonical 文件推理 adapter 共用这里的语义：长音频按 mic
原始帧数切换到 CPU、迁移失败时成组回滚 model/stft。
"""

from __future__ import annotations

import math

import torch


def validate_cpu_long_audio_seconds(cpu_long_audio_seconds: float) -> float:
    """校验 AEC 长音频 CPU 阈值（秒）。"""
    threshold = float(cpu_long_audio_seconds)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("cpu_long_audio_seconds must be finite and non-negative")
    return threshold


def resolve_aec_file_device(
    requested_device,
    num_samples: int,
    sample_rate: int,
    cpu_long_audio_seconds: float,
) -> torch.device:
    """按 mic 原始帧数决定当前 AEC 文件的设备。

    阈值为 0 表示关闭自动切换；启用时仅当原始时长严格大于阈值才切到 CPU。
    """
    requested = torch.device(requested_device)
    threshold = validate_cpu_long_audio_seconds(cpu_long_audio_seconds)
    if requested.type == "cpu" or threshold == 0:
        return requested
    if int(num_samples) > threshold * int(sample_rate):
        return torch.device("cpu")
    return requested


def move_aec_runtime(model, stft_module, target_device, rollback_device) -> None:
    """成组迁移 AEC runtime；任一异常时尽力恢复迁移前设备。"""
    target = torch.device(target_device)
    rollback = torch.device(rollback_device)
    try:
        model.to(target)
        stft_module.to(target)
    except BaseException as move_error:
        rollback_errors = []
        for component in (model, stft_module):
            try:
                component.to(rollback)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                f"AEC device migration failed ({move_error}); rollback failed: "
                f"{rollback_errors!r}"
            ) from move_error
        raise
