"""已物化 offline 多通道数据的 geometry 与通道排列合同。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.data.simulation.array_profiles import ResolvedMicrophoneArray, resolve_microphone_array


def resolve_offline_spatial_geometry(
    *,
    spatial_enabled: bool,
    array_profile: str | None,
    custom_geometry: Any,
    reference_channel: int,
    channel_order: Sequence[Any] | None,
) -> tuple[ResolvedMicrophoneArray | None, tuple[Any, ...] | None]:
    """解析 split geometry；空间 split 未配置 geometry 时立即失败。

    ``channel_order`` 是磁盘数据排列的声明，可使用物理通道 ID（例如
    ``("left", "right")``），不要求是 ``0..C-1``，也不会触发重排。
    """
    if type(spatial_enabled) is not bool:
        raise TypeError("spatial_enabled must be an exact bool")
    if spatial_enabled and array_profile is None and custom_geometry is None:
        raise ValueError(
            "offline spatial split requires array_profile or custom_geometry; "
            "microphone count is derived from geometry"
        )
    resolved = resolve_microphone_array(
        array_profile=array_profile,
        custom_geometry=custom_geometry,
        reference_channel=reference_channel,
    )
    if not spatial_enabled:
        if channel_order is not None:
            raise ValueError("channel_order is only valid when spatial_enabled=True")
        return None, None
    if resolved is None:
        raise ValueError("spatial_enabled=True requires resolved microphone geometry")
    if channel_order is None:
        raise ValueError("spatial_enabled=True requires channel_order")
    if isinstance(channel_order, (str, bytes)):
        raise TypeError("channel_order must be a sequence")
    order = tuple(channel_order)
    if len(order) != resolved.channels or len(set(order)) != len(order):
        raise ValueError(
            f"channel_order must contain exactly C={resolved.channels} unique entries"
        )
    return resolved, order


def validate_offline_audio_channels(path: str, expected_channels: int, *, source: str) -> None:
    """用音频 header 校验实际 C，不从首个文件推断 split geometry。"""
    import soundfile as sf

    info = sf.info(path)
    actual = int(info.channels)
    if actual != expected_channels:
        raise ValueError(
            f"{source} has {actual} audio channels, expected geometry C={expected_channels}: {path}"
        )
